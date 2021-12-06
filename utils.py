"""
Utils
:author: Qizhi Li
"""
import os
import torch
import numpy as np
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import time
import pickle
import json
import pyprind

from static_data import file_path as fp


class Utils:
    def __init__(self):
        pass

    def read_file(self, type_, path):
        """
        Loading file, include:
            1. pickle
            2. json

        :param type_: str
                The type of file
        :param path: str
                File path
        :return data: Object
                The data in the file
        """
        if type_ == 'pkl':
            with open(path, 'rb') as f:
                data = pickle.load(f)
        elif type_ == 'json':
            with open(path, 'r') as f:
                data = json.load(f)
        else:
            return None

        return data

    def write_file(self, type_, path, data):
        """
        Writing file, include:
            1. pickle
            2. json

        :param type_: str
                The type of file
        :param path: str
                File path
        :param data: Object
                The data need to write into the file
        """
        if type_ == 'pkl':
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        elif type_ == 'json':
            with open(path, 'w') as f:
                json.dump(data, f)
        elif type_ == 'txt_w':
            with open(path, 'w') as f:
                f.writelines(data)

    def load_glove_embed(self, vocab):
        """
        Encoding each word in the vocab into GloVe embeddings
        :param vocab: list
                ['word1', 'word2', 'word3', ..., 'wordn']
        :return embed: LongTensor
        """
        embed = torch.LongTensor(len(vocab), 300)

        # In order to ensure the word order, it is necessary to read out the words
        # in the lexicon first, and then encode them as word embeddings
        glove = {}
        pper = pyprind.ProgPercent(len(vocab), monitor=True)
        with open(fp.glove_embed, encoding='utf-8') as f:
            for line in f:
                word = line[0:line.index(' ')]
                if word in vocab or word == 'UNK':
                    strvec = line[line.index(' ') + 1:]
                    glove[word] = torch.LongTensor([float(x) for x in strvec.split(' ')])
                    pper.update()

        i = 0
        for word in vocab:
            try:
                embed[i] = glove[word]
            except KeyError:
                embed[i] = glove['UNK']
            i += 1

        return embed

    def load_w2v_wv(self):
        """
        Loading Word2Vec word embeddings
        :return w2v: Object
        """
        print('========== Start loading w2v ==========')
        start = time.time()
        w2v = KeyedVectors.load_word2vec_format(fp.word2vec_embed, binary=True)
        print('w2v loaded, it took %.2f sec.' % (time.time() - start))
        return w2v

    def get_wv_embed(self, vocab, wv='w2v'):
        """
        Obtaining word embeddings
        :param vocab: Iterator (set / list)
                The list need to do the mapping operation
        :param wv: str
                'glove': obtain GloVe word embeddings
                'w2v': obtain Word2Vec word embeddings
                default: 'w2v'
        :return embed: tensor
        """
        if wv == 'w2v':
            if os.path.exists(fp.word2vec_embed_pkl):
                word_embed = self.read_file('pkl', fp.word2vec_embed_pkl)
            else:
                word_embed = self.load_w2v_wv()
                self.write_file('pkl', fp.word2vec_embed_pkl, word_embed)

            embed = torch.FloatTensor(len(vocab), 300)
            i = 0
            for word in vocab:
                try:
                    embed[i] = torch.from_numpy(word_embed[word].copy())
                except KeyError:
                    embed[i] = torch.from_numpy(word_embed['UNK'].copy())
                i += 1
        elif wv == 'glove':
            if os.path.exists(fp.glove_embed_pkl):
                embed = self.read_file('pkl', fp.glove_embed_pkl)
            else:
                embed = self.load_glove_embed(vocab)
                self.write_file('pkl', fp.glove_embed_pkl, embed)
        else:
            print('Please choose "glove" or "w2v"')
            quit()

        return embed

    def get_id_word_mapping(self, X_set):
        """
        Obtain the mapping relationships between the word and the id
        :param X_set: set
                The set of the words
        :return word2idx: dict
                The mapping relationships between the word an the id
        :return idx2word: dict
                The mapping relationships between the id an the word
        """
        pad = '<PAD>'
        word2idx = {}
        idx2word = {}
        for i, word in enumerate(X_set):
            word2idx[word] = i
            idx2word[i] = word

        # Add <PAD> to the last digit of the list
        idx2word[len(idx2word)] = pad
        word2idx[pad] = len(word2idx)

        return word2idx, idx2word

    def get_sentence_idx(self, X_sentences, word2idx, max_seq_length):
        """
        获得句子中每个词语的映射关系
        Obtaining the mapping relationships between the the word and id in one sentence
        :param X_sentences: list
                [['word11', 'word12', ..., 'word1n'],
                 ['word21', 'word22', ..., 'word2m'],
                 ...,
                 ['wordN1', 'wordN2', ..., 'wordNk']]
        :param word2idx: dict
                The mapping relationships between the word an the id
        :param max_seq_length: int
        :return sentences_idx: LongTensor
                [[idx11, idx12, ..., idx1n],
                 [idx21, idx22, ..., idx2m],
                 ...,
                 [idxN1, idxN2, ..., idxNk]]
        """
        sentences_ids = []

        for sentence in X_sentences:
            word_count = 0
            sentence_ids = []
            for word in sentence:
                sentence_ids.append(word2idx[word])
                word_count += 1
                # If the number of words exceeds max_seq_length, the excess part is truncated
                if word_count >= max_seq_length:
                    break
            # If the number of words is less than max_seq_length,
            # fill in <PAD> to max_seq_length
            if word_count < max_seq_length:
                sentence_ids.extend([word2idx['<PAD>']] * (max_seq_length - word_count))

            sentences_ids.append(sentence_ids)

        return torch.LongTensor(sentences_ids)

    def get_word_embed(self, X_set, wv='w2v'):
        """
        Obtaining word embeddings
        :param X_set: set
                Word set
        :param wv: str
                'glove': obtaining GloVe word embeddings
                'w2v': obtaining Word2Vec word embeddings
                default: 'w2v'
        :return embed: FloatTensor
                All word embeddings in SemEval dataset or SST
        """
        embed = self.get_wv_embed(X_set, wv=wv)
        return embed
