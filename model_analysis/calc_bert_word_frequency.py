"""
Finding the word frequency of all words in the dataset
:author: Qizhi Li
"""
import argparse
import pyprind
from collections import Counter

from utils import Utils
from static_data import file_path as fp
from preprocess.load_experiment_dataset import load_data


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', help='semeval or SST', type=str)
args = parser.parse_args()

utils = Utils()
enhanced_words = list(utils.read_file('json', fp.bert_vec_need_to_enhance).values())
word_frequencies = {}

X_train, _, X_dev, _, X_test, _ = load_data(args.dataset)
if args.dataset == 'semeval':
    word2idx = utils.read_file('json', fp.semeval_word2idx)
    idx2word = utils.read_file('json', fp.semeval_idx2word)
else:
    word2idx = utils.read_file('json', fp.sst_word2idx)
    idx2word = utils.read_file('json', fp.sst_idx2word)

Xs = []
Xs.extend(X_train)
Xs.extend(X_dev)
Xs.extend(X_test)

X_ids = []
enhanced_words_ids = []

oov = 0
for enhanced_word in enhanced_words:
    try:
        enhanced_words_ids.append(word2idx[enhanced_word])
    except KeyError:
        oov += 1

print("There are %d words outside the vocabulary in the 'enhanced_words'" % oov)

# Since word2idx is generated from the dataset, we don't have to use 'try' to
# judge whether the word is inside the vocab
for X in Xs:
    sentence_ids = []
    for word in X:
        sentence_ids.append(word2idx[word])
    X_ids.append(sentence_ids)

pper = pyprind.ProgPercent(len(X_ids))
for sentence in X_ids:
    for X_id in sentence:
        if X_id in enhanced_words_ids:
            try:
                word_frequencies[idx2word[str(X_id)]] += 1
            except KeyError:
                word_frequencies[idx2word[str(X_id)]] = 1
            pper.update()
        else:
            pper.update()
            continue

most_common = Counter(word_frequencies).most_common()