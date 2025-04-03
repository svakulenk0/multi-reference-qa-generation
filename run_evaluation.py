'''

Specify the dataset, decoding approach, number of sequences and samples for evaluation:
    python run_evaluation.py --generation greedy --nsequences 3 --nsamples 1 --verbose True
    python run_evaluation.py --generation topn_marginal --nsequences 10
    python run_evaluation.py --generation topn_marginal --nsequences 3 --checkpoint uniform_2-epochs

or run to reproduce the evaluation on NewsQA:
    python run_evaluation.py > out.log

'''
import sys
import argparse
from collections import defaultdict

import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import T5TokenizerFast, T5ForConditionalGeneration
from transformers import BeamSearchScorer, StoppingCriteriaList, MaxLengthCriteria

from sentence_transformers import SentenceTransformer, util

from rouge_score import rouge_scorer
from datasets import load_metric


# for rouge computation
sys.setrecursionlimit(8735 * 2080 + 10)

NBEAMS = 5
threshold = 0.1  # for marginal decoding

model_name = 't5-base'
max_source_length = 512
max_target_length = 1024
max_length = max_target_length

top_k = 50
top_p = 0.92

model_path = './models/%s/'

datasets = ['newsqa', 'monserrate']
newsqa_path = './data/NewsQA/entity-driven-benchmark-test-dedup.csv'
monserrate_path = './data/MONSERRATE_Corpus.txt'


checkpoints = ['baseline_2-epochs', 'standard_2-epochs', 'uniform_2-epochs']
decoding = ['greedy', 'beam', 'topk', 'topp', 'topn', 'topn_marginal']               
               

def distinct_n_sentence_level(references, tokenizer):
    '''
    Distinct token metric
    '''
    tokens = [t for r in references for t in set(tokenizer.tokenize(r))]
    if tokens:
        return len(set(tokens)) / len(tokens)


def run(dataset, checkpoint_name, generation, nsequences,
        nsamples, verbose, num_beams=NBEAMS):
    '''
    checkpoint_name -- model checkpoint
    generation -- decoding approach
    '''    
    assert dataset in datasets
    
    if generation.split('_')[0] == 'topn':
        batch_size = 1
        stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=max_target_length)])
    else:
        batch_size = 8 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    tokenizer = T5TokenizerFast.from_pretrained(model_name, model_max_length=max_target_length)
    model = T5ForConditionalGeneration.from_pretrained(model_path%checkpoint_name)
    
    # load data
    data_path = newsqa_path
    df = pd.read_csv(data_path, sep=',', header=None)

    sentence_ids = df[0].tolist()
    context = df[2].tolist()
    answers = df[3].tolist()
    questions = df[4].tolist()
                
    sent2qa = defaultdict(dict)
    input_sequences = []
    for i, s in enumerate(sentence_ids):
        if s not in sent2qa:
            sent2qa[s]['answers'] = []
            sent2qa[s]['questions'] = []
            input_sequences.append(context[i])
        sent2qa[s]['answers'].append(answers[i])
        sent2qa[s]['questions'].append(questions[i])
    
    sent_ids = list(sent2qa.keys())

    # trim dataset
    if nsamples != -1:
        input_sequences = input_sequences[:nsamples]
        sent_ids = sent_ids[:nsamples]
    
    encoding = tokenizer(input_sequences, 
                         padding='longest', 
                         max_length=max_source_length, 
                         truncation=True, 
                         return_tensors="pt")
    input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

    test_data = TensorDataset(input_ids, attention_mask, torch.Tensor(sent_ids))
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    model.to(device)
    model.eval()

    # initialize evaluation metrics
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    sbert_model = SentenceTransformer('distilbert-base-nli-stsb-quora-ranking')
    
    rougel_f, rougel_fq = [], []
    distinct_as, distinct_qs = [], []
    num_qa_per_s = []
    softmatcheda05, softmatcheda08 = 0, 0
    softmatchedq05, softmatchedq08 = 0, 0
    sbert_scores = []
    
    for step, batch in enumerate(test_dataloader):
#         print(step, 'Step')
        
        # Load batch to GPU
        with torch.no_grad():
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
            
            if generation == 'greedy':
                outputs = model.generate(input_ids=b_input_ids, attention_mask=b_attn_mask,
                                         max_length=max_target_length)
                
            # generation: beam search
            elif generation == 'beam':
                if num_beams < nsequences:
                    num_beams = nsequences

                outputs = model.generate(input_ids=b_input_ids,
                                         attention_mask=b_attn_mask,
                                         num_beams=num_beams, 
    #                                      early_stopping=True,
    #                                      no_repeat_ngram_size=2,
                                         num_return_sequences=nsequences,
                                         max_length=max_target_length)

            # generation: topk
            elif generation == 'topk':
                outputs = model.generate(input_ids=b_input_ids,
                                         attention_mask=b_attn_mask,
                                         do_sample=True,
                                         num_return_sequences=nsequences,
                                         top_k=top_k,
                                         max_length=max_target_length)

            elif generation == 'topp':
                outputs = model.generate(input_ids=b_input_ids,
                                         attention_mask=b_attn_mask,
                                         do_sample=True,
                                         num_return_sequences=nsequences,
                                         top_p=top_p, 
                                         top_k=0,
                                         max_length=max_target_length)

            elif generation.split('_')[0] == 'topn':
                
                # produce outputs with generate and decode to predictions
                encoder_outputs = model.encoder(b_input_ids, #attention_mask=b_attn_mask,
                                        return_dict=True, output_hidden_states=True)

                decoder_input = [model.config.decoder_start_token_id] + [3, 9, 3155]


                start_ids = torch.tensor([decoder_input]*batch_size).to(model.device)

        #         print(encoder_outputs)

                # sample first tokens for the n-ary answer set
                output = model(decoder_input_ids=start_ids, encoder_outputs=encoder_outputs)


                next_token_logits = output.logits[:, -1, :]
                next_token_scores = nn.functional.softmax(next_token_logits, dim=-1)
                next_token_scores, next_tokens = torch.topk(
                        next_token_scores, nsequences, dim=1, largest=True, sorted=True
                    )
                

                outputs = []
                for i in range(nsequences):

                    # stop iterating if difference to the next token exceeds the threshold
                    if generation == 'topn_marginal':
                        if i!=0 and (next_token_scores[0][i].item() / next_token_scores[0][i-1].item()) < threshold:
                            break

                    next_token = next_tokens[:, i]

                    next_token = next_token[:, None]
                    decoder_input_ids = torch.cat([start_ids, next_token], dim=-1)

                    cur_len = 1
                    while True:

                        # sequence restart
                        if cur_len == 1:
                            pkv = output.past_key_values

                        o = model(decoder_input_ids=next_token, encoder_outputs=encoder_outputs,
                                  return_dict=True, output_hidden_states=True, past_key_values=pkv)
                        pkv = o.past_key_values

                        next_tokens_scores = o.logits[:, -1, :]
                        next_token = torch.argmax(next_tokens_scores, dim=-1)

                        # update generated ids, model inputs, and length for next step
                        next_token = next_token[:, None]
                        decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=-1)

                        cur_len = cur_len + 1
                        if cur_len == max_target_length or next_token[0].item() == 0:
                            break

                    decoder_input_ids = decoder_input_ids.tolist()[0]
                    outputs.append(decoder_input_ids)

            # decode
            prediction = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            sentences = tokenizer.batch_decode(b_input_ids, skip_special_tokens=True)

        for i, s in enumerate(sentences):
            # gt
            s_id = int(b_labels[i].item())
            gt_answers = sent2qa[s_id]['answers']
            gt_questions = sent2qa[s_id]['questions']
            
            # parse output sequences
            if generation == 'greedy':
                predicted = [prediction[i]]
            else:
                predicted = prediction[nsequences*i:nsequences*i+nsequences]
            
            if verbose:
                print(s)
                print("GT:")
                for i in range(len(gt_answers)):
                    print("a> %s q> %s" %(gt_answers[i], gt_questions[i]))
                print('Predicted:')
                
            answers, questions = [], []
            for p in predicted:
                if verbose:
                    print(p)
                if 'q>' in p:
                    splits = p.split('q>')
                    a = splits[0]
                    q = splits[1].lstrip()
                else:
                    q = ''
                    a = p.strip()
                if 'a>' in a:
                    a = a.split('a>')[-1].strip()
                questions.append(q)
                answers.append(a)

            num_qa_per_s.append(len(questions))

            # measure diversity of the generated answers
            d_metrics = distinct_n_sentence_level(answers, tokenizer)    
            if d_metrics:
                distinct_as.append(d_metrics)
        
            # check how well we can recall each of the gt_answers
            for t in gt_answers:
                score_dicts = [scorer.score(t, a) for a in answers]
                max_score = {}
                for k in scorer.rouge_types:
                    index = np.argmax([s[k].fmeasure for s in score_dicts])
                    max_score[k] = score_dicts[index][k]
                rougel_f.append(max_score['rougeL'].fmeasure)
                
                if max_score['rougeL'].fmeasure > 0.5:
                    softmatcheda05 += 1
                    if max_score['rougeL'].fmeasure > 0.8:
                        softmatcheda08 += 1

            # measure diversity of the generated questions
            d_metrics = distinct_n_sentence_level(questions, tokenizer)    
            if d_metrics:
                distinct_qs.append(distinct_n_sentence_level(questions, tokenizer))
            
            #Compute embedding for both lists
            embeddings_qgt = sbert_model.encode(gt_questions, convert_to_tensor=True)
            embeddings_qpr = sbert_model.encode(questions, convert_to_tensor=True)
            
            #Compute cosine-similarities
            cosine_scores = util.cos_sim(embeddings_qgt, embeddings_qpr)
            max_scores = torch.max(cosine_scores, axis=1).values.tolist()
            sbert_scores.extend(max_scores)
            if verbose:
                print(cosine_scores)
            
            # check how well we can recall each of the gt_questions
            for t in gt_questions:
                score_dicts = [scorer.score(t, q) for q in questions]
                max_score = {}
                for k in scorer.rouge_types:
                    index = np.argmax([s[k].fmeasure for s in score_dicts])
                    max_score[k] = score_dicts[index][k]
                if verbose:
                    print(t)
                    print(max_score['rougeL'].fmeasure)
                    print(max_score['rougeL'].fmeasure)
                rougel_fq.append(max_score['rougeL'].fmeasure)

                if max_score['rougeL'].fmeasure > 0.5:
                    softmatchedq05 += 1
                    if max_score['rougeL'].fmeasure > 0.8:
                        softmatchedq08 += 1
                
    print(len(rougel_fq), 'samples')
    print('# qa per sentence', np.mean(num_qa_per_s))

    print('Answers:')
    print('distinct %.2f' % np.mean(distinct_as))
    print('rougel_f mean %.2f' % np.mean(rougel_f))
    print('rougel_f var %.2f' % np.var(rougel_f))

    print('softm05 %.2f' % (softmatcheda05/len(rougel_fq)))
    print('softm08 %.2f' % (softmatcheda08/len(rougel_fq)))


    print('Questions:')
    print('distinct %.2f' % np.mean(distinct_qs))
    print('rougel_f mean %.2f' % np.mean(rougel_fq))
    print('rougel_f var %.2f' % np.var(rougel_fq))
    print('softm05 %.2f' % (softmatchedq05/len(rougel_fq)))
    print('softm08 %.2f' % (softmatchedq08/len(rougel_fq)))
    print('sbert %.2f' % np.mean(sbert_scores))


def run_eval(dataset, checkpoint_name, generation, nsequences,
             nsamples, verbose):
    
    print(dataset, '\n')

    if checkpoint_name:
        run(dataset, checkpoint_name, generation, nsequences,
            nsamples, verbose)
    else:
        for c in checkpoints:
            
            if generation:
                print(c)
                run(dataset, c, generation, nsequences,
                    nsamples, verbose)
            else:
                for d in decoding:
                    print(c)
                    print(d)
                    run(dataset, c, d, nsequences,
                        nsamples, verbose)
                    print('\n')
            print('\n')
            print('<>')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs='?', default='newsqa')
    parser.add_argument("--generation", nargs='?', default=None)
    parser.add_argument("--checkpoint", nargs='?', default=None)
    parser.add_argument('--nsequences', type=int, nargs='?', default=3)
    parser.add_argument('--nsamples', type=int, nargs='?', default=5000)
    parser.add_argument('--verbose', type=bool, nargs='?', default=False)

    args = parser.parse_args()
    
    run_eval(args.dataset, args.checkpoint, args.generation,
             args.nsequences, args.nsamples, args.verbose)
