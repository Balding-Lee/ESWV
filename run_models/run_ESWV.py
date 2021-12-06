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
from random import choice
from transformers.models.bert import BertTokenizer, BertModel
import argparse

sys.path.append('..')
from static_data import file_path as fp
from utils import Utils
from models.ESWV import ESWV


def get_bert_embeddings(final_lexicon, utils):
    """
    Obtain the word embedding of BERT
    :param final_lexicon: json
    :param utils: Object
    :return embeddings: LongTensor
            shape: (num_word_in_bert, 768)
    :return X: LongTensor
            shape: (num_word_in_bert)
    :return: LongTensor
            shape: (num_word_in_bert)
    :return id_word_mapping: dict
            The mapping relationships between id and word in BERT
    :return idx2id_mapping: dict
            The mapping relationships between index and id in BERT
    """
    model = BertModel.from_pretrained(fp.bert_base_uncased)
    tokenizer = BertTokenizer.from_pretrained(fp.bert_base_uncased)
    bert_embed = model.get_input_embeddings()

    words = list(final_lexicon.keys())
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
        y.append(final_lexicon[word])

    # Since using id directly will cause IndexError,
    # we need to do a new mapping between id and index
    idx2id_mapping = {}
    for idx, id_ in enumerate(id_word_mapping.keys()):
        idx2id_mapping[idx] = id_

    X = torch.tensor(list(idx2id_mapping.keys()))
    embeddings = bert_embed(torch.tensor(list(id_word_mapping.keys())))

    if not os.path.exists(fp.bert_vec_need_to_enhance):
        utils.write_file('json', fp.bert_vec_need_to_enhance, id_word_mapping)

    return embeddings, X, torch.tensor(y), id_word_mapping, idx2id_mapping


def get_word_id_mapping_and_y(final_lexicon):
    """
    Obtaining the mapping relationships between id and word, and the true labels
    :param final_lexicon: dict
    :return word2idx: dict
            The mapping relationships between word and id
    :return idx2word: dict
            The mapping relationships between id and word
    :return y: tensor
            True labels
    """
    word2idx = {}
    idx2word = {}
    y = []
    i = 0
    for k in final_lexicon.keys():
        word2idx[k] = i
        idx2word[i] = k
        i += 1
        y.append(final_lexicon[k])

    return word2idx, idx2word, torch.LongTensor(y)


def get_embed_X_y(utils, wv='w2v'):
    """
    Obtaining data iter
    :param utils: Object
    :param wv: str
            'glove': obtain GloVe word embeddings
            'w2v': obtain Word2Vec word embeddings
    :return embed: tensor
            The vector of Embed layer
    :return X: LongTensor
            texts
    :return y: LongTensor
            labels
    :return idx2word: dict
            The mapping relationships between id and word
    """
    final_lexicon = utils.read_file('json', fp.final_lexicon)
    word2idx, idx2word, y = get_word_id_mapping_and_y(final_lexicon)
    embed = utils.get_wv_embed(list(word2idx.keys()), wv=wv)
    X = torch.LongTensor(list(idx2word.keys()))

    return embed, X, y, idx2word


def generate_corrupted_sample(idx2word, final_lexicon):
    """
    Generating corrupted sample.

    Extracte a random word in vocab as corrupted X, and ensure that the sentiment
    orientation of the corrupted sample is different from the original sentiment
    orientation.
    :param idx2word: dict
            The mapping relationships between id and word
    :param final_lexicon: dict
    :return corrupted_X: LongTensor
            The id of corrupted sample
    :return corrupted_y: LongTensor
            The sentiment orientation of corrupted sample
    """
    orientation_list = [0, 1, 2]
    corrupted_X = choice(list(idx2word.keys()))
    corrupted_y = choice(orientation_list)

    # If the randomly selected sentiment orientation is the same as the original
    # sentiment orientation, then randomly selected it again
    while corrupted_y == final_lexicon[idx2word[corrupted_X]]:
        corrupted_y = choice(orientation_list)

    return torch.tensor(corrupted_X).reshape(-1), torch.tensor(corrupted_y).reshape(-1)


def generate_bert_corrupted_sample(id_word_mapping, idx2id_mapping, final_lexicon):
    """
    Generating corrupted sample of BERT
    :param id_word_mapping: dict
            The mapping relationships between id and word of BERT
    :param idx2id_mapping: dict
            The mapping relationships between index and id of BERT
    :param final_lexicon: dict
    :return corrupted_X: LongTensor
            The id of corrupted sample
    :return corrupted_y: LongTensor
            The sentiment orientation of corrupted sample
    """
    orientation_list = [0, 1, 2]
    corrupted_X = choice(list(idx2id_mapping.keys()))
    corrupted_y = choice(orientation_list)

    # If the randomly selected sentiment orientation is the same as the original
    # sentiment orientation, then randomly selected it again
    # idx2id_mapping[corrupted_X]: id
    # id_word_mapping[id]: word
    # final_lexicon[word]: orientation
    while corrupted_y == final_lexicon[id_word_mapping[idx2id_mapping[corrupted_X]]]:
        corrupted_y = choice(orientation_list)

    return torch.tensor(corrupted_X).reshape(-1), torch.tensor(corrupted_y).reshape(-1)


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
            num_layers:
                0: no hidden layer
                1: one hidden layer
                2: two hidden layers
            vector:
                'BERT': train ESWV with BERT word embeddings
                'w2v': train ESWV with Word2Vec word embeddings
                'glove': train ESWV with GloVe word embeddings
    :return:
    """
    utils = Utils()
    final_lexicon = utils.read_file('json', fp.final_lexicon)

    embed, X, y= None, None, None
    save_path = '%sESWV_%s_hidden%s_parameters.pkl' % (fp.embeddings, args.vector,
                                                       args.num_layers)
    if args.vector == 'w2v':
        embed, X, y, idx2word = get_embed_X_y(utils, wv='w2v')
    elif args.vector == 'glove':
        embed, X, y, idx2word = get_embed_X_y(utils, wv='glove')
    elif args.vector == 'BERT':
        embed, X, y, id_word_mapping, idx2id_mapping = get_bert_embeddings(final_lexicon,
                                                                           utils)

    assert embed is not None
    assert X is not None
    assert y is not None
    assert 0 <= args.num_layers <= 2

    X_train, X_dev, X_test, y_train, y_dev, y_test = train_dev_test_split(X, y)
    train_dataset = Data.TensorDataset(X_train, y_train)
    dev_dataset = Data.TensorDataset(X_dev, y_dev)
    test_dataset = Data.TensorDataset(X_test, y_test)
    train_iter = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_iter = Data.DataLoader(dev_dataset, batch_size=batch_size)
    test_iter = Data.DataLoader(test_dataset, batch_size=batch_size)

    model = ESWV(embed, 3, args).to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dev_best_loss = float('inf')
    # Record the iter of batch that the loss of the last validation set dropped
    last_improve = 0
    # Whether the result has not improved for a long time
    flag = False
    require_improvement = args.early_stop
    i = 0
    for epoch in range(num_epochs):

        model.train()
        for X, y in train_iter:
            if args.vector == 'w2v' or args.vector == 'glove':
                corrupted_X, corrupted_y = generate_corrupted_sample(idx2word,
                                                                     final_lexicon)
            else:
                corrupted_X, corrupted_y = generate_bert_corrupted_sample(id_word_mapping,
                                                                          idx2id_mapping,
                                                                          final_lexicon)
            X = torch.cat((X, corrupted_X)).to(device)
            y = torch.cat((y, corrupted_y)).to(device)
            y_hat = model(X)
            optimizer.zero_grad()
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                pred = torch.max(y_hat.data, 1)[1].cpu().numpy()
                acc = accuracy_score(y.detach().cpu().numpy(), pred)
                f1 = f1_score(y.detach().cpu().numpy(), pred, average='macro')

                dev_loss, dev_acc, dev_f1 = evaluate(model, dev_iter, device)
                print('iter %d, train loss %f, train accuracy %f, train macro-F1 %f,'
                      'dev loss %f, dev accuracy %f, dev macro-F1 %f' %
                      (i + 1, l.item(), acc, f1, dev_loss, dev_acc, dev_f1))

                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
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

    test_loss, test_acc, test_f1 = evaluate(model, test_iter, device)
    print('test loss %f, test accuracy %f, test macro-F1 %f' %
          (test_loss, test_acc, test_f1))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument('-l', '--num_layers', help='number of hidden layers, 0, 1, or 2',
                    default=0, type=int)
parser.add_argument('-v', '--vector', help='w2v, glove, or BERT')
parser.add_argument('-b', '--batch_size', type=int)
parser.add_argument('-e', '--early_stop', type=int)
parser.add_argument('-lr', '--learning_rate', type=float)
parser.add_argument('-d', '--dropout', type=float)
args = parser.parse_args()
train(num_epochs=200, batch_size=args.batch_size, lr=args.learning_rate,
      device=device, args=args)
