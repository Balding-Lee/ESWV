"""
Running the trained model on the test dataset
:author: Qizhi Li
"""
import sys
import torch
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import argparse
import warnings

sys.path.append('..')
from preprocess import load_experiment_dataset
from static_data import file_path as fp
from utils import Utils


def enhance_word_embed(embed, args):
    """
    增强词向量
    :param embed: tensor
    :param args: Object
            vector:
                'w2v': loading senti_vec trained by Word2Vec
                'glove': loading senti_vec trained by GloVe
            num_layers:
                0: ESWV with no hidden layer
                1: ESWV with one hidden layer
                2: ESWV with two hidden layers
    :return ESWV_embed: tensor
            enhanced sentiment embeddings
    """
    file_path = '%sESWV_%s_hidden%s_parameters.pkl' % (fp.embeddings, args.vector,
                                                       args.num_layers)
    ESWV_params = torch.load(file_path)

    senti_vec = ESWV_params['senti_vec'].reshape(-1)
    ESWV_embed = torch.add(embed, senti_vec)

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


def load_BERT_data(batch_size, args):
    """
    :param utils: Object
    :param batch_size: int
    :param args: Object
            dataset: semeval or SST
    :return:
    """
    X_train, y_train, X_dev, y_dev, X_test, y_test = load_experiment_dataset.load_data(args.dataset,
                                                                                       False)
    test_batch_X, test_batch_y, test_batch_count = get_data_iter(X_test,
                                                                 y_test,
                                                                 batch_size)

    return test_batch_X, test_batch_y, test_batch_count


def get_data_iter(X, y, batch_size):
    """
    :param X: ndarray
            texts
    :param y: ndarray
            labels
    :param batch_size: int
    :return batch_X: list
            [ndarray, ndarray, ..., ndarray]
            putting the texts into a list according to the batch
    :return batch_y: list
            [ndarray, ndarray, ..., ndarray]
            putting the labels into a list according to the batch
    :return batch_count: int
            how many batches are there in the data
    """
    batch_count = int(len(X) / batch_size)
    batch_X, batch_y = [], []
    for i in range(batch_count):
        batch_X.append(X[i * batch_size: (i + 1) * batch_size])
        batch_y.append(y[i * batch_size: (i + 1) * batch_size])

    return batch_X, batch_y, batch_count


def load_others_data(utils, batch_size, args):
    """
    :param utils: Object
    :param batch_size: int
    :param args: Object
            dataset: semeval or SST
    :return X_set: set
            The set of all words
    :return test_iter: Object
    :return max_seq_length: int
    """
    max_seq_length = 0
    X_train, y_train, X_dev, y_dev, X_test, y_test = load_experiment_dataset.load_data(args.dataset,
                                                                                       True)
    X_set = get_word_set(X_train, X_dev, X_test)
    word2idx = None
    if args.dataset == 'semeval':
        word2idx = utils.read_file('json', fp.semeval_word2idx)
        max_seq_length = 130

    elif args.dataset == 'SST':
        word2idx = utils.read_file('json', fp.sst_word2idx)
        max_seq_length = 100

    assert max_seq_length != 0
    assert word2idx is not None

    X_test = utils.get_sentence_idx(X_test, word2idx, max_seq_length)
    y_test = torch.LongTensor(y_test)

    test_dataset = Data.TensorDataset(X_test, y_test)

    test_iter = Data.DataLoader(test_dataset, batch_size)

    return X_set, test_iter, max_seq_length


def evaluate_others(model, test_iter, device):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    model.cpu()

    with torch.no_grad():
        for texts, labels in test_iter:
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

    print('loss %f, accuracy %f, macro-F1 %f' % (loss_total / len(test_iter), acc, macro_F1))


def evaluate_BERT(model, X_test, y_test, batch_count, device):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    model.to(device)

    with torch.no_grad():
        for i in range(batch_count):
            inputs = X_test[i]
            labels = torch.tensor(y_test[i]).to(device)

            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = accuracy_score(labels_all, predict_all)
    macro_F1 = f1_score(labels_all, predict_all, average='macro')

    print('loss %f, accuracy %f, macro-F1 %f' % (loss_total / len(X_test), acc, macro_F1))


def test_bert(args, device):
    model, batch_size = None, 64

    if args.type_of_vec == 'original':
        model_path = '%s%s_%s_%s_round%d.pkl' % (fp.experiment_results,
                                                 args.model,
                                                 args.dataset,
                                                 args.type_of_vec,
                                                 args.round)
    else:
        model_path = '%s%s_%s_%s_round%s_hidden%s.pkl' % (fp.experiment_results,
                                                          args.model,
                                                          args.dataset,
                                                          args.type_of_vec,
                                                          args.round, args.num_layers)

    model = torch.load(model_path)

    test_batch_X, test_batch_y, test_batch_count = load_BERT_data(batch_size, args)

    evaluate_BERT(model, test_batch_X, test_batch_y, test_batch_count, device)


def test_others(args, utils, device):
    model, batch_size = None, 64

    if args.type_of_vec == 'original':
        model_path = '%s%s_%s_%s_%s.pkl' % (fp.experiment_results, args.model, args.dataset,
                                            args.vector, args.type_of_vec)
    else:
        model_path = '%s%s_%s_%s_%s_hidden%s.pkl' % (fp.experiment_results, args.model,
                                                     args.dataset, args.vector,
                                                     args.type_of_vec, args.num_layers)

    if args.model == 'TextRCNN' or args.model == 'TextBiRCNN':
        batch_size = 128

    X_set, test_iter, max_seq_length = load_others_data(utils, batch_size, args)
    model = torch.load(model_path)

    model.eval()

    evaluate_others(model, test_iter, device)


utils = Utils()
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', help='semeval or SST', type=str)
parser.add_argument('-v', '--vector', help="w2v or glove", type=str)
parser.add_argument('-m', '--model',
                    help="TextCNN, BiLSTM, TextRCNN, TextBiRCNN, GRU, BiGRU, BERT", type=str)
parser.add_argument('-t', '--type_of_vec', help='original or enhanced')
parser.add_argument('-r', '--round', help='1, 2, or 3', default=None, type=int)
parser.add_argument('-l', '--num_layers', help='0, 1, or 2', default=0, type=int)
args = parser.parse_args()

assert 0 <= args.num_layers <= 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.model == 'BERT':
    test_bert(args, device)
else:
    test_others(args, utils, device)
