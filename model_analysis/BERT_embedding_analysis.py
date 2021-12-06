"""
Analyze BERT word embeddings
:author: Qizhi Li
"""
import sys
import torch
from transformers.models.bert import BertConfig, BertForSequenceClassification
import argparse

sys.path.append('..')
from static_data import file_path as fp
from utils import Utils


def get_bert_mean_embed(args, utils):
    """
    Obtain the average value of all the BERT word embeddings, include:
        1. original word embeddings;
        2. enhanced word embeddings;
        3. trained original word embeddings;
        4. trained enhanced word embeddings.
    :param args: Object
            'num_layers': how many hidden layers of ESWV
            'round': which round of running results
            'dataset': which dataset you choose
    :param utils: Object
    :return bert_base_mean_embed: tensor
            The mean vector of all BERT original word embeddings
    :return bert_enhanced_mean_embed: tensor
            The mean vector of all BERT enhanced word embeddings
    :return trained_original_mean_embed: tensor
            The mean vector of all trained BERT original word embeddings
    :return trained_enhanced_mean_embed: tensor
            The mean vector of all trained BERT enhanced word embeddings
    """
    pretrained_weights = fp.bert_base_uncased
    config = BertConfig.from_pretrained(pretrained_weights, num_labels=2)
    bert_base = BertForSequenceClassification.from_pretrained(pretrained_weights,
                                                              config=config)
    bert_base_embed = bert_base.get_input_embeddings().weight.data
    bert_base_mean_embed = bert_base_embed.mean(0)

    id_word_mapping = utils.read_file('json', fp.bert_vec_need_to_enhance)
    ESEV_file_path = '%sESWV_BERT_hidden%s_parameters.pkl' % (fp.embeddings,
                                                              args.num_layers)
    ESWV_params = torch.load(ESEV_file_path)
    senti_vec = ESWV_params['senti_vec'].reshape(-1)
    for id_ in id_word_mapping.keys():
        bert_base_embed[int(id_)] = torch.add(bert_base_embed[int(id_)], senti_vec)

    bert_enhanced_mean_embed = bert_base_embed.mean(0)

    bert_trained_original_file_path = '%sBERT_%s_original_round%s.pkl' % (fp.experiment_results,
                                                                          args.dataset,
                                                                          args.round)
    bert_trained_enhanced_file_path = '%sBERT_%s_enhanced_round%s_hidden%s.pkl' % (fp.experiment_results,
                                                                                   args.dataset,
                                                                                   args.round,
                                                                                   args.num_layers)
    bert_trained_original = torch.load(bert_trained_original_file_path, map_location='cpu')
    bert_trained_enhanced = torch.load(bert_trained_enhanced_file_path, map_location='cpu')
    trained_original_embed = bert_trained_original.bert.get_input_embeddings().weight.data
    trained_enhanced_embed = bert_trained_enhanced.bert.get_input_embeddings().weight.data

    trained_original_mean_embed = trained_original_embed.mean(0)
    trained_enhanced_mean_embed = trained_enhanced_embed.mean(0)

    return bert_base_mean_embed, bert_enhanced_mean_embed, trained_original_mean_embed, trained_enhanced_mean_embed


def get_bert_word_embed(args, utils):
    """
    Obtain the word embedding of a word in BERT, include:
        1. original word embedding;
        2. enhanced word embedding;
        3. trained original word embedding;
        4. trained enhanced word embedding.
    :param args: Object
            'word': word need to analyze
            'num_layers': how many hidden layers of ESWV
            'round': which round of running results
            'dataset': which dataset you choose
    :param utils: Object
    :return original_embed: tensor
            original BERT word embedding
    :return enhanced_embed: tensor
            enhanced BERT word embedding
    :return trained_original_word_embed: tensor
            trained original BERT word embedding
    :return trained_enhanced_word_embed: tensor
            trained enhanced BERT word embedding
    """
    word = args.word
    pretrained_weights = fp.bert_base_uncased
    config = BertConfig.from_pretrained(pretrained_weights, num_labels=2)
    bert_base = BertForSequenceClassification.from_pretrained(pretrained_weights,
                                                              config=config)
    bert_base_embed = bert_base.get_input_embeddings().weight.data

    id_word_mapping = utils.read_file('json', fp.bert_vec_need_to_enhance)
    # Finding the key of a given word
    word_id = int(list(id_word_mapping.keys())[list(id_word_mapping.values()).index(word)])

    # Obtaining ESWV vectors
    ESEV_file_path = '%sESWV_BERT_hidden%s_parameters.pkl' % (fp.embeddings,
                                                              args.num_layers)
    ESWV_params = torch.load(ESEV_file_path)
    senti_vec = ESWV_params['senti_vec'].reshape(-1)

    original_embed = bert_base_embed[word_id]
    enhanced_embed = torch.add(original_embed, senti_vec)

    # Loading the trained word embeddings
    bert_trained_original_file_path = '%sBERT_%s_original_round%s.pkl' % (fp.experiment_results,
                                                                          args.dataset,
                                                                          args.round)
    bert_trained_enhanced_file_path = '%sBERT_%s_enhanced_round%s_hidden%s.pkl' % (fp.experiment_results,
                                                                                   args.dataset,
                                                                                   args.round,
                                                                                   args.num_layers)
    bert_trained_original = torch.load(bert_trained_original_file_path, map_location='cpu')
    bert_trained_enhanced = torch.load(bert_trained_enhanced_file_path, map_location='cpu')
    trained_original_embed = bert_trained_original.bert.get_input_embeddings().weight.data
    trained_enhanced_embed = bert_trained_enhanced.bert.get_input_embeddings().weight.data

    trained_original_word_embed = trained_original_embed[word_id]
    trained_enhanced_word_embed = trained_enhanced_embed[word_id]

    return original_embed, enhanced_embed, trained_original_word_embed, trained_enhanced_word_embed


utils = Utils()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', help='semeval or SST', type=str)
parser.add_argument('-w', '--word', help='the word you need', type=str)
parser.add_argument('-r', '--round', help='1, 2, or 3', default=None, type=int)
parser.add_argument('-l', '--num_layers', help='0, 1, or 2', default=0, type=int)
parser.add_argument('-m', '--mode', help='word: word level embed, mean: mean embed',
                    type=str, default='mean')
args = parser.parse_args()

if args.mode == 'word':
    oe, ee, towe, tewe = get_bert_word_embed(args, utils)
    word_embed = oe.reshape(-1, oe.shape[0])
    word_embed = torch.cat((word_embed, ee.reshape(-1, ee.shape[0])), dim=0)
    word_embed = torch.cat((word_embed, towe.reshape(-1, towe.shape[0])), dim=0)
    word_embed = torch.cat((word_embed, tewe.reshape(-1, tewe.shape[0])), dim=0)

    print('==================== %s ====================' % args.word)
    print('The cosine similarity between the original vector and the enhanced vector is %f' %
            torch.cosine_similarity(word_embed[0], word_embed[1], dim=0))
    print('The cosine similarity between the trained original vector and the trained enhanced vector is %f' %
            torch.cosine_similarity(word_embed[2], word_embed[3], dim=0))
else:
    ome, eme, tome, teme = get_bert_mean_embed(args, utils)
    print('The cosine similarity between the mean of the original vector and '
          'the mean of the enhanced vector is %f' % torch.cosine_similarity(ome, eme, dim=0))
    print('The cosine similarity between the mean of the trained original vector and '
          'the mean of the trained enhanced vector is %f' % torch.cosine_similarity(tome, teme, dim=0))

# draw_fig(original_embed, enhanced_embed, trained_original_embed, trained_enhanced_embed)
