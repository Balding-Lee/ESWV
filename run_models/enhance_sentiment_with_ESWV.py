"""
Enhancing sentiment orientation of data by ESWV
:author: Qizhi Li
"""
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import time
import argparse
import warnings

sys.path.append('..')
from static_data import file_path as fp
from models.CNN import CNN
from models.LSTM import LSTM
from models.TextRCNN import TextRCNN
from preprocess import load_experiment_dataset
from utils import Utils


def enhance_word_embed(X_set, utils, embed, args):
    """
    :param X_set: set
            词语集合
    :param utils: Object
    :param embed: tensor
            word embeddings
    :param args: Object
            vector:
                'w2v': loading senti_vec trained by Word2Vec
                'glove': loading senti_vec trained by GloVe
    :return ESWV_embed: tensor
            enhanced word embeddings
    """
    if args.enhance_mode == 'l':
        file_path = '%sESWV_%s_parameters.pkl' % (fp.embeddings, args.vector)
        lexicon_word2idx = utils.read_file('json', fp.final_lexicon_word2idx)
    else:
        file_path = '%sESWV_%s_%s_parameters.pkl' % (fp.embeddings,
                                                     args.vector,
                                                     args.dataset)
        if args.dataset == 'semeval':
            lexicon_word2idx = utils.read_file('json', fp.semeval_lexicon_word2idx)
        else:
            lexicon_word2idx = utils.read_file('json', fp.sst_lexicon_word2idx)

    ESWV_params = torch.load(file_path, map_location='cpu')

    senti_vec = ESWV_params['senti_vec.weight']
    X_list = list(X_set)

    ESWV_embed = embed.clone()

    for word in lexicon_word2idx.keys():
        if word in X_list:
            X_id = X_list.index(word)
            ESWV_embed[X_id] = torch.add(ESWV_embed[X_id],
                                         senti_vec[lexicon_word2idx[word]])


    return ESWV_embed


def get_word_list(X_list):
    """
    Combining all words together to form a list
    :param X_list: list
            [['word11', 'word12', ..., 'word1n'],
             ['word21', 'word22', ..., 'word2m'],
             ...,
             ['wordN1', 'wordN2', ..., 'wordNk']]
    :return word_list: list
            ['word11', 'word12', ..., 'word1n', 'word21', ..., 'wordNk']
    """
    word_list = []
    for sentence in X_list:
        for word in sentence:
            word_list.append(word)

    return word_list


def get_word_set(X_train, X_dev, X_test):
    """
    Obtaining word set
    :param X_train: list
    :param X_dev: list
    :param X_test: list
    :return: set
    """
    train_list = get_word_list(X_train)
    dev_list = get_word_list(X_dev)
    test_list = get_word_list(X_test)

    return set(train_list + dev_list + test_list)


def load_data(utils, args):
    """
    :param utils: Object
    :param args: Object
            dataset: semeval or SST
    :return X_set: set
            The set of all words
    :return train_iter: Object
    :return dev_iter: Object
    :return test_iter: Object
    :return max_seq_length: int
    """
    max_seq_length = 0
    X_train, y_train, X_dev, y_dev, X_test, y_test = load_experiment_dataset.load_data(args.dataset,
                                                                                       True)
    X_set = get_word_set(X_train, X_dev, X_test)
    if args.dataset == 'semeval':
        max_seq_length = 130

    elif args.dataset == 'SST':
        max_seq_length = 100

    assert max_seq_length != 0

    # Saving the id for test
    if args.dataset == 'semeval':
        if os.path.exists(fp.semeval_word2idx):
            word2idx = utils.read_file('json', fp.semeval_word2idx)
            idx2word = utils.read_file('json', fp.semeval_idx2word)
        else:
            word2idx, idx2word = utils.get_id_word_mapping(X_set)
            utils.write_file('json', fp.semeval_word2idx, word2idx)
            utils.write_file('json', fp.semeval_idx2word, idx2word)
    else:
        if os.path.exists(fp.sst_word2idx):
            word2idx = utils.read_file('json', fp.sst_word2idx)
            idx2word = utils.read_file('json', fp.sst_idx2word)
        else:
            word2idx, idx2word = utils.get_id_word_mapping(X_set)
            utils.write_file('json', fp.sst_word2idx, word2idx)
            utils.write_file('json', fp.sst_idx2word, idx2word)

    X_train = utils.get_sentence_idx(X_train, word2idx, max_seq_length)
    X_dev = utils.get_sentence_idx(X_dev, word2idx, max_seq_length)
    X_test = utils.get_sentence_idx(X_test, word2idx, max_seq_length)
    y_train = torch.LongTensor(y_train)
    y_dev = torch.LongTensor(y_dev)
    y_test = torch.LongTensor(y_test)

    train_dataset = Data.TensorDataset(X_train, y_train)
    dev_dataset = Data.TensorDataset(X_dev, y_dev)
    test_dataset = Data.TensorDataset(X_test, y_test)

    train_iter = Data.DataLoader(train_dataset, args.batch_size, shuffle=True)
    dev_iter = Data.DataLoader(dev_dataset, args.batch_size)
    test_iter = Data.DataLoader(test_dataset, args.batch_size)

    return X_set, train_iter, dev_iter, test_iter, max_seq_length


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
    macro_F1 = f1_score(labels_all, predict_all, average='macro')
    return loss_total / len(data_iter), acc, macro_F1


def train(num_epochs, device, args, enhanced=True):
    num_outputs = 2
    utils = Utils()
    X_set, train_iter, dev_iter, test_iter, max_seq_length = load_data(utils, args)

    save_path = '%s%s_%s_%s' % (fp.experiment_results, args.model, args.dataset, args.vector)
    embed = utils.get_word_embed(X_set, wv=args.vector)

    if enhanced:
        ESWV_embed = enhance_word_embed(X_set, utils, embed, args)
        embed = torch.cat((embed, torch.zeros(1, ESWV_embed.shape[1])))
        if args.enhance_mode == 'l':
            save_path += '_enhanced.pkl'
        else:
            save_path += '_enhanced_c.pkl'
    else:
        # The vector of <PAD> is all 0s
        embed = torch.cat((embed, torch.zeros(1, embed.shape[1])))
        save_path += '_original.pkl'

    model = None
    if args.model == 'TextCNN':
        model = CNN(embed, num_outputs, max_seq_length).to(device)
    elif args.model == 'TextRCNN':
        model = TextRCNN(embed, 256, False, num_outputs, max_seq_length)
    elif args.model == 'TextBiRCNN':
        model = TextRCNN(embed, 256, True, num_outputs, max_seq_length)
    elif args.model == 'BiLSTM':
        model = LSTM(embed, embed.shape[1], 128, num_outputs, bidirectional=True)

    assert model is not None

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss = nn.CrossEntropyLoss()

    # If there are multiple GPU, set the data on witch GPU
    model_device = next(model.parameters()).device

    require_improvement = args.early_stop
    dev_best_loss = float('inf')
    # Record the iter of batch that the loss of the last validation set dropped
    last_improve = 0
    # Whether the result has not improved for a long time
    flag = False
    i = 0
    start = time.time()
    for epoch in range(num_epochs):
        model.train()
        for X, y in train_iter:
            X = X.to(model_device)
            y = y.to(model_device)

            y_hat = model(X)
            optimizer.zero_grad()
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                pred = torch.max(y_hat.data.cpu(), 1)[1].numpy()
                acc = accuracy_score(y.detach().cpu().numpy(), pred)
                macro_F1 = f1_score(y.detach().cpu().numpy(), pred, average='macro')

                dev_loss, dev_acc, dev_macro_F1 = evaluate(model, dev_iter, device)

                model.train()

                print('iter %d, train loss %f, train accuracy %f, train macro-F1 %f, '
                      'dev loss %f, dev accuracy %f, dev macro-F1 %f' % (
                          i + 1, l.item(), acc, macro_F1, dev_loss, dev_acc, dev_macro_F1
                      ))

                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model, save_path)
                    last_improve = i
                model.to(model_device)

            if i - last_improve > require_improvement:
                # Stop training if the loss of dev dataset has not dropped
                # exceeds args.early_stop batches
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
            i += 1
        if flag:
            break

    print('%.2f seconds used' % (time.time() - start))

    model = torch.load(save_path)
    test_loss, test_acc, test_macro_F1 = evaluate(model, test_iter, device)
    print('test loss %f, test accuracy %f, test macro-F1 %f' % (
        test_loss, test_acc, test_macro_F1))


warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', help='semeval or SST', type=str)
parser.add_argument('-v', '--vector', help="w2v or glove", type=str)
parser.add_argument('-m', '--model', help="TextCNN, BiLSTM, TextRCNN, TextBiRCNN", type=str)
parser.add_argument('-t', '--type_of_vec', help='original or enhanced')
parser.add_argument('-b', '--batch_size', default=64, type=int)
parser.add_argument('-e', '--early_stop', default=256, type=int)
parser.add_argument('-em', '--enhance_mode', default='l', type=str,
                    help='only train lexicon (l) or train all the corpus (c)')
args = parser.parse_args()

assert args.enhance_mode == 'l' or args.enhance_mode == 'c'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if args.type_of_vec == 'original':
    train(40, device, args, enhanced=False)
elif args.type_of_vec == 'enhanced':
    train(40, device, args, enhanced=True)
