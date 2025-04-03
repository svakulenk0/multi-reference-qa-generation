import argparse

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import T5TokenizerFast, T5ForConditionalGeneration
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback, IntervalStrategy


path = './models/baseline_2-epochs'  # aqg-joint_newsqa_t5-base_2-epochs' # best aqg
model_name = 't5-base'
max_source_length = 512
max_target_length = 1024

voc_size = 32128
limit = 80000 # 38105  # 29024
test_size = 512
batch_size = 8


def create_distribution(seqs, tokenizer):
    labels = []
    for i in seqs:
        indices = tokenizer.convert_tokens_to_ids(i)
        pmass = 1 / len(indices)
        distr = np.zeros(voc_size)
        distr[indices] = pmass
        labels.append(distr)
    return torch.tensor(labels)


def dummy_data_collector(features):
    batch = {}
    batch['input_ids'] = torch.stack([f[0] for f in features])
    batch['attention_mask'] = torch.stack([f[1] for f in features])
    batch['labels'] = torch.stack([f[2] for f in features])
    
    return batch


class MultitaskDataloader:
    """
    Data loader that combines and samples from multiple single-task
    data loaders.
    """

    def __init__(self, dataloader_dict):
        self.dataloader_dict = dataloader_dict
        self.num_batches_dict = {
            task_name: len(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        self.task_name_list = list(self.dataloader_dict)
        self.dataset = [None] * sum(
            len(dataloader.dataset) for dataloader in self.dataloader_dict.values()
        )

    def __len__(self):
        return sum(self.num_batches_dict.values())

    def __iter__(self):
        """
        For each batch, sample a task, and yield a batch from the respective
        task Dataloader.
        We use size-proportional sampling, but you could easily modify this
        to sample from some-other distribution.
        """
        task_choice_list = []
        for i, task_name in enumerate(self.task_name_list):
            task_choice_list += [i] * self.num_batches_dict[task_name]
        task_choice_list = np.array(task_choice_list)
        np.random.shuffle(task_choice_list)
        dataloader_iter_dict = {
            task_name: iter(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        for task_choice in task_choice_list:
            task_name = self.task_name_list[task_choice]
            yield next(dataloader_iter_dict[task_name])


class CustomTrainer(Trainer):
    
    def get_single_train_dataloader(self, task_name, train_dataset):
        """
        Create a single-task data loader that also yields task names
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_sampler = (
            RandomSampler(train_dataset)
            if self.args.local_rank == -1
            else DistributedSampler(train_dataset)
        )

        data_loader = DataLoader(
                train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=train_sampler,
                collate_fn=self.data_collator,
            )
        return data_loader

    def get_train_dataloader(self):
        """
        Returns a MultitaskDataloader, which is not actually a Dataloader
        but an iterable that returns a generator that samples from each
        task Dataloader
        """
        return MultitaskDataloader(
            {
                task_name: self.get_single_train_dataloader(task_name, task_dataset)
                for task_name, task_dataset in self.train_dataset.items()
            }
        )
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        input_ids = inputs.get("input_ids")
        model = model.module
        
        # check the labels supplied are probability distribution or output sequences
        if labels.size(-1) == voc_size:
            decoder_input = [model.config.decoder_start_token_id] + [3, 9, 3155]
            decoder_input_ids = torch.tensor([decoder_input] * batch_size).to(model.device)

            outputs = model(input_ids=input_ids.to(model.device), decoder_input_ids=decoder_input_ids)
            logits = outputs[0]
            # last predicted
            logits = logits[:, -1, :]
            labels = labels.to(model.device)
        else:
            outputs = model(**inputs)
            logits = outputs.get("logits")
            labels = labels.view(-1)
        
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels)
        
        return (loss, outputs) if return_outputs else loss


def run_eval(mode, epochs):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer =  T5TokenizerFast.from_pretrained(model_name, model_max_length=1024)

    model = T5ForConditionalGeneration.from_pretrained(path)
    model.to(device)

    # load data
    df = pd.read_csv('./data/NewsQA/entity-driven-benchmark-train.csv', sep=',', header=None)
    df.head()

    # prepare input and output
    context = df[2].tolist()
    answers = df[3].tolist()
    questions = df[4].tolist()
    
    c2as = {}
    for i, c in enumerate(context):
        if c not in c2as:
            c2as[c] = []
        if isinstance(answers[i], str):
            # tokenize string
            tokens = tokenizer.tokenize(answers[i])
            token = tokens[0]
            if token == '▁':
                token = tokens[1]
            if token not in c2as[c]:
                c2as[c].append(token) # 'a> ' + 

    # create splits
    input_sequences = list(c2as.keys())[:limit]
    output_sequences = list(c2as.values())[:limit]

    # shuffle and split into train/validation sets
    input_sequences, input_dev, output_sequences, output_dev =\
        train_test_split(input_sequences, output_sequences, test_size=test_size, random_state=2020)

    nsamples = len(input_sequences)
    
    if mode == 'uniform':
        # encode inputs
        encoding = tokenizer(input_sequences, 
                             padding='longest', 
                             max_length=max_source_length, 
                             truncation=True, 
                             return_tensors="pt")
        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask


        encoding = tokenizer(input_dev, 
                             padding='longest', 
                             max_length=max_source_length, 
                             truncation=True, 
                             return_tensors="pt")
        input_ids_dev, attention_mask_dev = encoding.input_ids, encoding.attention_mask


        labels = create_distribution(output_sequences, tokenizer)
        labels_dev = create_distribution(output_dev, tokenizer)


        train_data_distr = TensorDataset(input_ids, attention_mask, labels)
        
        model_label = 'uniform_%d-epochs' % epochs
    else:
        model_label = 'standard_%d-epochs' % epochs

    input_sequences = context[:nsamples]
    output_sequences = []
    for i, question in enumerate(questions):
        output_sequences.append('a> %s q> %s' % (answers[i], questions[i]))
        if len(output_sequences) == nsamples:
            break

    print(len(output_sequences), 'samples in total')
    assert(len(output_sequences) == len(input_sequences))

    # shuffle and split into train/validation sets
    input_sequences, input_dev, output_sequences, output_dev =\
        train_test_split(input_sequences, output_sequences, test_size=test_size, random_state=2020)

    print(len(input_sequences), 'training samples')
    print(len(input_dev), 'dev samples')

    encoding = tokenizer(input_sequences, 
                         padding='longest', 
                         max_length=max_source_length, 
                         truncation=True, 
                         return_tensors="pt")
    input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

    # encode the targets
    target_encoding = tokenizer(output_sequences, 
                                padding='longest', 
                                max_length=max_target_length, 
                                truncation=True, 
                                return_tensors="pt")
    labels = target_encoding.input_ids

    encoding = tokenizer(input_dev, 
                         padding='longest', 
                         max_length=max_source_length, 
                         truncation=True, 
                         return_tensors="pt")
    input_ids_dev, attention_mask_dev = encoding.input_ids, encoding.attention_mask

    target_encoding = tokenizer(output_dev, 
                                padding='longest', 
                                max_length=max_target_length, 
                                truncation=True, 
                                return_tensors="pt")
    labels_dev = target_encoding.input_ids

    # Create the DataLoader for our training set
    train_data = TensorDataset(input_ids, attention_mask, labels)
    dev_data = TensorDataset(input_ids_dev, attention_mask_dev, labels_dev)

    training_args = TrainingArguments(
                                output_dir="./results/" + model_label, # output directory
                                num_train_epochs=epochs, # total # of training epochs
                                per_device_train_batch_size=1, # batch size per device during training
                                per_device_eval_batch_size=1, # batch size for evaluation
                                warmup_steps=100, # number of warmup steps for learning rate scheduler
                                weight_decay=0.01, # strength of weight decay
                                save_total_limit=5, # Only last x models are saved. Older ones are deleted.
                                evaluation_strategy = IntervalStrategy.STEPS,
                                eval_steps=500, # Evaluation and Save happens every x steps
                                load_best_model_at_end=True,
                                metric_for_best_model="eval_loss")


    if mode == 'uniform':
        
        train_dataset = {
            'original': train_data,
            'distr': train_data_distr,
        }
                
        trainer = CustomTrainer(model=model, args=training_args, train_dataset=train_dataset,
                                data_collator=dummy_data_collector, eval_dataset=dev_data,
                                callbacks = [EarlyStoppingCallback(early_stopping_patience=3)])

    else:        
        trainer = Trainer(model=model, args=training_args, train_dataset=train_data,
                          data_collator=dummy_data_collector, eval_dataset=dev_data,
                          callbacks = [EarlyStoppingCallback(early_stopping_patience=5)])


    trainer.train()


    model.save_pretrained('./models/%s/' % model_label)
    print('./models/%s/' % model_label)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("mode")
    parser.add_argument("--epochs")

    args = parser.parse_args()
    
    run_eval(args.mode, int(args.epochs))
