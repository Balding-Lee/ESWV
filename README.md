# ESWV

Source code and dataset for `ESWV`

## Requirements:

+ Python >= 3.7
+ PyTorch >= 1.6.0
+ pyprind
+ transformers >= 4.9.2

## Notice:

Since the fine-trained models, and ESWV parameters files are too large, we put them in https://pan.baidu.com/s/1oWC3F0MDU2EH0RVd14YCsw with the extraction code `dmcr`.

Just download the folder, and then copy the three `subfolders` in the folder to the `ESWV project folder`.

Due to the poor code reusability of our code, you may find that similar code will appear in some two files.

## Usage:

### Reproduce the trained model on the test set:

If you have downloaded the `experiment_results` folder and put it into the project, just use the following commands to reproduce the trained model on the test set.

```python
cd run_models

# dataset: semeval
# model: TextBiRCNN
# word embeddings: w2v
# word emebddings type: original
python run_trained_model_on_test.py -d semeval -m TextBiRCNN -v w2v -t original

# dataset: SST
# model: TextCNN
# word embeddings: glove
# word embeddings type: enhanced
# ESWV hidden layers: 2
python run_trained_model_on_test.py -d SST -m TextCNN -v glove -t enhanced -l 2


# -d --dataset: semeval or SST
# -m --model: TextCNN, BiLSTM, TextRCNN, TextBiRCNN
# -v --vector: w2v or glove
# -t --type_of_vec: original or enhanced
# -l --num_layers: 0, 1, or 2  default=0    # only enhanced needs
```

The above commands are only part of the demonstration. If you want to reproduce all the experiments, please follow the last comment to set the command.

### Training models with fine-trained ESWV:

Please make sure that you have downloaded `embeddings` folder and put it into the project. You can use the following commands to train TextCNN, BiLSTM, TextRCNN, and TextBiRCNN from scratch.

```python
cd run_models

# dataset: semeval
# model: TextCNN
# word embeddings: w2v
# word embeddings type: original
python enhance_sentiment_with_ESWV.py -d semeval -m TextCNN -v w2v -t original

# dataset: SST
# model: TextRCNN
# word embeddings: glove
# word embeddings type: enhanced
# ESWV hidden layers: 0
# batch size: 128
# early stop: 512
python enhance_sentiment_with_ESWV.py -d SST -m TextRCNN -v glove -t enhanced -l 0 -b 128 -e 512

The above commands are only part of the demonstration. If you want to train all the models with fine-trained ESWV from scratch, please follow the last comments to set the command.

### Training ESWV

You can use the following commands to train ESWV from scratch. The hyper-parameters setting is in `embeddings > parameters_setting`.

```python
cd run_models

# no hidden layer
python run_ESWV.py -l 0 -v w2v -b 31 -e 512 -lr 1e-3 -d 0.8
python run_ESWV.py -l 1 -v w2v -b 15 -e 256 -lr 5e-5 -d 0.2
python run_ESWV.py -l 2 -v w2v -b 15 -e 128 -lr 1e-4 -d 0.19
# 1 hidden layer
python run_ESWV.py -l 0 -v glove -b 31 -e 512 -lr 1e-3 -d 0.8
python run_ESWV.py -l 1 -v glove -b 15 -e 256 -lr 5e-5 -d 0.2
python run_ESWV.py -l 2 -v glove -b 15 -e 128 -lr 1e-4 -d 0.19

# -l --num_layers: 0, 1, or 2
# -v --vector: w2v, glove
```

### Others

If you want to get the final lexicon, use the following command.

```python
cd preprocess

python get_ANEW_SC_lexicon.py
```



If you want to get the max_seq_length, use the following command.

```python
cd preprocess

python get_max_seq_length.py
```

This command will not tell you witch max_seq_length is appropriate, but just show you two image about word frequencies. Please decide which parameter is appropriate based on your own observation.



If you want to find the most common word in the corpus, use the following command.

```python
cd model_analysis

python calc_bert_word_frequency.py -d semeval
python calc_bert_word_frequency.py -d SST
```

This command will not tell you which word is suitable for the experiment too, but just give you one `dict` with words and their frequency. Please choose them by your own observation.





