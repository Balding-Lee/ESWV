"""
Runing ESWV
:author: Qizhi Li
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from transformers.models.bert import BertTokenizer, BertModel
import argparse

sys.path.append('..')
from static_data import file_path as fp
from utils import Utils
from models.ESWV import ESWV


def get_bert_embeddings(sentiment_lexicon, utils, args):
    """
    Obtain the word embedding of BERT
    :param sentiment_lexicon: json
    :param utils: Object
    :return embeddings: LongTensor
            shape: (num_word_in_bert, 768)
    :return X: LongTensor
            shape: (num_word_in_bert)
    :return: LongTensor
            shape: (num_word_in_bert)
    :return vocab_size: int
            The size of the vocab
    """
    model = BertModel.from_pretrained(fp.bert_base_uncased)
    tokenizer = BertTokenizer.from_pretrained(fp.bert_base_uncased)
    bert_embed = model.get_input_embeddings()

    words = list(sentiment_lexicon.keys())
    UNK_id = tokenizer.unk_token_id

    word_id_mapping = {}
    id_word_mapping = {}
    word_not_in_embed = 0
    for word in words:
        token_id = tokenizer.convert_tokens_to_ids(word)
        if token_id == UNK_id:
            word_not_in_embed += 1
            continue
        else:
            word_id_mapping[word] = token_id
            id_word_mapping[token_id] = word

    print('The number of words in the lexicon but not in the word embedding of BERT is %d'
          % word_not_in_embed)

    y = []
    for word in word_id_mapping.keys():
        y.append(sentiment_lexicon[word])

    # Since using id directly will cause IndexError,
    # we need to do a new mapping between id and index
    idx2id_mapping = {}
    id2idx_mapping = {}
    for idx, id_ in enumerate(id_word_mapping.keys()):
        idx2id_mapping[idx] = id_
        id2idx_mapping[id_] = idx

    X = torch.tensor(list(idx2id_mapping.keys()))
    embeddings = bert_embed(torch.tensor(list(id_word_mapping.keys())))

    if args.enhance_mode == 'l':
        utils.write_file('json', fp.bert_lexicon_word2idx, word_id_mapping)
        utils.write_file('json', fp.bert_lexicon_idx2word, id_word_mapping)
        utils.write_file('json', fp.bert_senti2token, idx2id_mapping)
        utils.write_file('json', fp.bert_token2senti, id2idx_mapping)
    else:
        if args.corpus == 'semeval':
            utils.write_file('json', fp.bert_semeval_lexicon_word2idx, word_id_mapping)
            utils.write_file('json', fp.bert_semeval_lexicon_idx2word, id_word_mapping)
            utils.write_file('json', fp.bert_semeval_senti2token, idx2id_mapping)
            utils.write_file('json', fp.bert_semeval_token2senti, id2idx_mapping)
        else:
            utils.write_file('json', fp.bert_sst_lexicon_word2idx, word_id_mapping)
            utils.write_file('json', fp.bert_sst_lexicon_idx2word, id_word_mapping)
            utils.write_file('json', fp.bert_sst_senti2token, idx2id_mapping)
            utils.write_file('json', fp.bert_sst_token2senti, id2idx_mapping)

    return embeddings, X, torch.tensor(y), len(id_word_mapping)


def get_word_id_mapping(final_lexicon):
    """
    Obtaining the mapping relationships between id and word
    :param final_lexicon: dict
    :return word2idx: dict
            The mapping relationships between word and id
    :return idx2word: dict
            The mapping relationships between id and word
    """
    word2idx = {}
    idx2word = {}
    i = 0
    for k in final_lexicon.keys():
        word2idx[k] = i
        idx2word[i] = k
        i += 1

    return word2idx, idx2word


def get_embed_X_y(utils, sentiment_lexicon, args):
    """
    Obtaining data iter
    :param utils: Object
    :param sentiment_lexicon: dict
            if enhance_mode == 'l', sentiment_lexicon is final_lexicon
            if enhance_mode == 'c', sentiment_lexicon is corpus_lexicon
    :param wv: str
            'glove': obtain GloVe word embeddings
            'w2v': obtain Word2Vec word embeddings
    :return embed: tensor
            The vector of Embed layer
    :return X: LongTensor
            texts
    :return y: LongTensor
            labels
    :return vocab_size: int
            The size of the vocab
    """
    if args.enhance_mode == 'l':
        if os.path.exists(fp.final_lexicon_word2idx):
            word2idx = utils.read_file('json', fp.final_lexicon_word2idx)
            idx2word = utils.read_file('json', fp.final_lexicon_idx2word)
        else:
            word2idx, idx2word = get_word_id_mapping(sentiment_lexicon)
            utils.write_file('json', fp.final_lexicon_word2idx, word2idx)
            utils.write_file('json', fp.final_lexicon_idx2word, idx2word)
    else:
        if args.corpus == 'semeval':
            if os.path.exists(fp.semeval_lexicon_word2idx):
                word2idx = utils.read_file('json', fp.semeval_lexicon_word2idx)
                idx2word = utils.read_file('json', fp.semeval_lexicon_idx2word)
            else:
                word2idx, idx2word = get_word_id_mapping(sentiment_lexicon)
                utils.write_file('json', fp.semeval_lexicon_word2idx, word2idx)
                utils.write_file('json', fp.semeval_lexicon_idx2word, idx2word)
        else:
            if os.path.exists(fp.sst_lexicon_word2idx):
                word2idx = utils.read_file('json', fp.sst_lexicon_word2idx)
                idx2word = utils.read_file('json', fp.sst_lexicon_idx2word)
            else:
                word2idx, idx2word = get_word_id_mapping(sentiment_lexicon)
                utils.write_file('json', fp.sst_lexicon_word2idx, word2idx)
                utils.write_file('json', fp.sst_lexicon_idx2word, idx2word)

    embed = utils.get_wv_embed(list(word2idx.keys()), wv=args.vector)
    # When you load the json file, the keys of idx2word will be str
    X = torch.LongTensor(list(word2idx.values()))
    y = torch.LongTensor(list(sentiment_lexicon.values()))

    return embed, X, y, len(idx2word)


def train_dev_test_split(X, y):
    """
    :param X: LongTensor
            texts
    :param y: LongTensor
            labels
    :return:
    """
    X_train, X_dt, y_train, y_dt = train_test_split(X, y, test_size=0.4, random_state=0,
                                                    stratify=y)
    X_dev, X_test, y_dev, y_test = train_test_split(X_dt, y_dt, test_size=0.5,
                                                    random_state=0, stratify=y_dt)

    return X_train, X_dev, X_test, y_train, y_dev, y_test


def evaluate(model, data_iter, device):
    """
    Evaluating model on dev and test, and outputting loss, accuracy and macro-F1
    :param model: Object
    :param data_iter: DataLoader
            dev or test
    :param device: Object
    :return: float
            total loss
    :return acc: float
            total accuracy
    :return macro_F1: float
            total macro-F1
    """
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    model.cpu()

    with torch.no_grad():
        for texts, labels in data_iter:
            texts.to(device)
            labels.to(device)

            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = accuracy_score(labels_all, predict_all)
    f1 = f1_score(labels_all, predict_all, average='macro')
    return loss_total / len(data_iter), acc, f1


def train(num_epochs, batch_size, lr, device, args):
    """
    :param num_epochs: int
    :param batch_size: int
    :param lr: float
            learning rate
    :param device: Object
    :param args: Object
            vector:
                'BERT': train ESWV with BERT word embeddings
                'w2v': train ESWV with Word2Vec word embeddings
                'glove': train ESWV with GloVe word embeddings
    :return:
    """
    utils = Utils()

    embed, X, y, vocab_size = None, None, None, None

    if args.enhance_mode == 'l':
        save_path = '%sESWV_%s_parameters.pkl' % (fp.embeddings, args.vector)
        sentiment_lexicon = utils.read_file('json', fp.final_lexicon)
    else:
        save_path = '%sESWV_%s_%s_parameters.pkl' % (
            fp.embeddings, args.vector, args.corpus)
        if args.corpus == 'semeval':
            sentiment_lexicon = utils.read_file('json', fp.semeval_lexicon)
        else:
            sentiment_lexicon = utils.read_file('json', fp.sst_lexicon)

    if args.vector == 'w2v' or args.vector == 'glove':
        embed, X, y, vocab_size = get_embed_X_y(utils, sentiment_lexicon, args)
    elif args.vector == 'BERT':
        embed, X, y, vocab_size = get_bert_embeddings(sentiment_lexicon, utils, args)

    assert embed is not None
    assert X is not None
    assert y is not None
    assert vocab_size is not None

    train_dataset = Data.TensorDataset(X, y)
    train_iter = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = ESWV(embed, 3, vocab_size).to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_best_loss = float('inf')
    # Record the iter of batch that the loss of the last validation set dropped
    last_improve = 0
    # Whether the result has not improved for a long time
    flag = False
    require_improvement = args.early_stop
    i = 0
    for epoch in range(num_epochs):

        model.train()
        for X, y in train_iter:

            X = X.to(device)
            y = y.to(device)

            y_hat = model(X)
            optimizer.zero_grad()
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                pred = torch.max(y_hat.data, 1)[1].cpu().numpy()
                acc = accuracy_score(y.detach().cpu().numpy(), pred)
                f1 = f1_score(y.detach().cpu().numpy(), pred, average='macro')

                print('iter %d, train loss %f, train accuracy %f, train macro-F1 %f' %
                      (i + 1, l.item(), acc, f1))

                if l.item() < train_best_loss:
                    train_best_loss = l.item()
                    torch.save(model.state_dict(), save_path)
                    last_improve = i

                model.train()
                model.to(device)

            if i - last_improve > require_improvement:
                # Stop training if the loss of dev dataset has not dropped
                # exceeds args.early_stop batches
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
            i += 1
        if flag:
            break

    model.load_state_dict(torch.load(save_path))
    # test_loss, test_acc, test_f1 = evaluate(model, test_iter, device)
    test_loss, test_acc, test_f1 = evaluate(model, train_iter, device)
    print('test loss %f, test accuracy %f, test macro-F1 %f' %
          (test_loss, test_acc, test_f1))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument('-v', '--vector', help='w2v, glove, or BERT')
parser.add_argument('-b', '--batch_size', type=int)
parser.add_argument('-e', '--early_stop', type=int)
parser.add_argument('-lr', '--learning_rate', type=float)
parser.add_argument('-em', '--enhance_mode', default='l', type=str,
                    help='only train lexicon (l) or train all the corpus (c)')
parser.add_argument('-c', '--corpus', type=str,
                    help='semeval or SST')
args = parser.parse_args()

assert args.enhance_mode == 'l' or args.enhance_mode == 'c'

train(num_epochs=200, batch_size=args.batch_size, lr=args.learning_rate,
      device=device, args=args)
