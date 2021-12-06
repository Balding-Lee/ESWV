import torch
from torch import nn
from transformers.models.bert import BertConfig, BertTokenizer, BertForSequenceClassification

from static_data import file_path as fp
from utils import Utils


def load_BERT_embeddings_and_enhance(model, utils, args):
    """
    Loading all word embeddings of BERT, and enhancing them with ESEV
    :param model: Object
    :param utils: Object
    :param args: Object

    :return model: Object
            The model enhanced by ESEV
    """
    bert_embed = model.get_input_embeddings().weight.data

    file_path = '%sESWV_BERT_hidden%s_parameters.pkl' % (fp.embeddings,
                                                         args.num_layers)
    ESWV_params = torch.load(file_path)
    senti_vec = ESWV_params['senti_vec'].reshape(-1)

    id_word_mapping = utils.read_file('json', fp.bert_vec_need_to_enhance)

    for id_ in id_word_mapping.keys():
        bert_embed[int(id_)] = torch.add(bert_embed[int(id_)], senti_vec)

    model.set_input_embeddings(nn.Embedding.from_pretrained(bert_embed))

    return model


class BertClassificationModel(nn.Module):
    def __init__(self, num_outputs, device, args):
        super(BertClassificationModel, self).__init__()
        model_class, tokenizer_class, pretrained_weights = (BertForSequenceClassification,
                                                            BertTokenizer,
                                                            fp.bert_base_uncased)
        utils = Utils()
        if args.dataset == 'semeval':
            self.max_seq_length = 130
        else:
            self.max_seq_length = 100
        self.device = device
        self.config = BertConfig.from_pretrained(pretrained_weights, num_labels=num_outputs)
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        self.bert = model_class.from_pretrained(pretrained_weights, config=self.config)

        # Using ESWV to enhance BERT embeddings
        if args.vector_type == 'enhanced':
            self.bert = load_BERT_embeddings_and_enhance(self.bert, utils, args)
            # Setting the gradient, otherwise there is no gradient
            self.bert.get_input_embeddings().requires_grad_(True)

        self.softmax = nn.Softmax()

    def forward(self, batch_sentences):
        batch_tokenized = self.tokenizer.batch_encode_plus(batch_sentences, add_special_tokens=True,
                                                           max_length=self.max_seq_length,
                                                           pad_to_max_length=True, truncation=True)
        input_ids = torch.tensor(batch_tokenized['input_ids']).to(self.device)
        attention_mask = torch.tensor(batch_tokenized['attention_mask']).to(self.device)
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        return outputs.logits

