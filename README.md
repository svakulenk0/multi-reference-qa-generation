## Multi-Reference QA Generation


Code for training and evaluation of the Transformer-based question generation model introduced in the ECAI 2023 paper [Uniform training and marginal decoding for multi-reference question-answer generation](https://ebooks.iospress.nl/doi/10.3233/FAIA230539)

If you find our paper, datasets or code useful, please reference this work:

```
@Inproceedings{Vakulenko2023,
 author = {Svitlana Vakulenko and Bill Byrne and Adrià de Gispert},
 title = {Uniform training and marginal decoding for multi-reference question-answer generation},
 year = {2023},
 booktitle = {ECAI 2023},
}
```

## Training

To train the models:
```
python fine_tune.py standard

python fine_tune.py uniform --epochs 10
```

## Evaluation

To evaluate the trained models specify the decoding approach and the model checkpoint for evaluation (see # Decoding and # Models for the available options):


```
python run_evaluation.py model_name decoding_approach 
```

e.g.

```
python run_evaluation.py uniform_mixed_entities_t5-base_10-epochs greedy
```


### Decoding settings

* greedy
* beam
* topk
* topp
* marginal_topn
* marginal_greedy

### Trained models

Download from: https://figshare.com/collections/multi-reference-qa-generation/7753064

* original NewsQA baseline:

	baseline_2-epochs

* baseline fine-tuned on entities (entities_answers_trainer.py):

	aqg_entities_t5-base_10-epochs

	standard_2-epochs

* baseline fine-tuned on entities + uniform answer tokens (fine_tune_mixed.py):

	uniform_mixed_entities_t5-base_10-epochs

	uniform_2-epochs


### Datasets

Download from: https://figshare.com/collections/multi-reference-qa-generation/7753064

* [NewsQA](https://github.com/Maluuba/newsqa) (generated entity-driven QA benchmark):

	'./data/NewsQA/entity-driven-benchmark-test-dedup.csv'

* [MONSERRATE corpus](https://github.com/hprodrig/MONSERRATE_Corpus)

	'./data/MONSERRATE_Corpus.txt'


## License Summary

The documentation is made available under the Creative Commons Attribution-ShareAlike 4.0 International License. See the LICENSE file.
