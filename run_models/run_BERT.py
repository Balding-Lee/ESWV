"""
Running BERT
:author: Qizhi Li
"""
import sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import time
import warnings
import argparse

sys.path.append('..')
from static_data import file_path as fp
from models.BERT import BertClassificationModel
from preprocess import load_experiment_dataset


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


def evaluate(model, batch_X, batch_y, batch_count, device):
    """
    Evaluating model on dev and test, and outputting loss, accuracy and macro-F1
    :param model: Object
    :param batch_X: list
            验证集或测试集的数据
            The texts in the dev or test
    :param batch_y: list
            The labels in the dev or test
    :param batch_count: int
            how many batches are there in the data
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

    with torch.no_grad():
        for i in range(batch_count):
            inputs = batch_X[i]
            labels = torch.tensor(batch_y[i]).to(device)

            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = accuracy_score(labels_all, predict_all)
    f1 = f1_score(labels_all, predict_all, average='macro')
    return loss_total / batch_count, acc, f1


def train(num_epochs, batch_size, lr, device, args):
    """
    :param num_epochs: int
    :param batch_size: int
    :param lr: float
            learning rate
    :param device: Object
    :param args: Object
            'type': original word embeddings or enhanced word embeddings
            'dataset': SST or semeval
    :return:
    """
    X_train, y_train, X_dev, y_dev, X_test, y_test = load_experiment_dataset.load_data(args.dataset,
                                                                                       False)

    model = BertClassificationModel(2, device, args).to(device)

    if args.vector_type == 'original':
        save_path = '%sBERT_%s_%s_round%s.pkl' % (fp.experiment_results, args.dataset,
                                                  args.vector_type, args.round)
    else:
        save_path = '%sBERT_%s_%s_round%s_hidden%s.pkl' % (fp.experiment_results,
                                                           args.dataset,
                                                           args.vector_type,
                                                           args.round, args.num_layers)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_batch_X, train_batch_y, train_batch_count = get_data_iter(X_train,
                                                                    y_train,
                                                                    batch_size)
    dev_batch_X, dev_batch_y, dev_batch_count = get_data_iter(X_dev,
                                                              y_dev,
                                                              batch_size)
    test_batch_X, test_batch_y, test_batch_count = get_data_iter(X_test,
                                                                 y_test,
                                                                 batch_size)
    require_improvement = 256
    dev_best_loss = float('inf')
    # Record the iter of batch that the loss of the last validation set dropped
    last_improve = 0
    # Whether the result has not improved for a long time
    flag = False
    n = 0
    start = time.time()
    for epoch in range(num_epochs):
        model.train()
        for i in range(train_batch_count):
            inputs = train_batch_X[i]
            labels = torch.tensor(train_batch_y[i]).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            l = loss(outputs, labels)
            l.backward()
            optimizer.step()

            if (n + 1) % 100 == 0:
                pred = torch.max(outputs.data, 1)[1].cpu().numpy()
                acc = accuracy_score(labels.detach().cpu().numpy(), pred)
                f1 = f1_score(labels.detach().cpu().numpy(), pred, average='macro')

                dev_loss, dev_acc, dev_f1 = evaluate(model, dev_batch_X, dev_batch_y,
                                                     dev_batch_count, device)
                model.train()
                print("iter %d, train loss %f, train accuracy %f, train macro-F1 %f,"
                      " dev loss %f, dev accuracy %f, dev macro-F1 %f" % (
                          (n + 1), l.item(), acc, f1, dev_loss, dev_acc, dev_f1))

                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model, save_path)
                    last_improve = n

            if n - last_improve > require_improvement:
                # Stop training if the loss of dev dataset has not dropped
                # exceeds 256 batches
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
            n += 1
        if flag:
            break

    print('%.2f seconds used' % (time.time() - start))

    test_loss, test_acc, test_f1 = evaluate(model, test_batch_X, test_batch_y,
                                            test_batch_count, device)
    print('test loss %f, test accuracy %f, test macro-F1 %f' % (test_loss, test_acc, test_f1))


warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--vector_type', help='original or enhanced')
parser.add_argument('-d', '--dataset', help='SST or semeval')
parser.add_argument('-r', '--round', help='1, 2, or 3', type=int)
parser.add_argument('-l', '--num_layers', help='0, 1, or 2', default=0, type=int)
args = parser.parse_args()

assert 0 <= args.num_layers <= 2

train(10, 64, 5e-6, device, args)
