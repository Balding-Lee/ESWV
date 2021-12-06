"""
TranS模型
:author: Qizhi Li
"""
import torch
import torch.nn as nn
from sklearn.decomposition import PCA


class ESWV(nn.Module):

    def __init__(self, embed, num_outputs, args):
        super().__init__()
        if args.num_layers == 0:
            dense_input = embed.shape[1]
        elif args.num_layers == 1:
            if args.vector == 'BERT':
                hidden1_output = 256
                dense_input = 256
            else:
                hidden1_output = 100
                dense_input = 100
            self.hidden1 = nn.Linear(embed.shape[1], hidden1_output)
        else:
            if args.vector == 'BERT':
                hidden1_output = 256
                hidden2_output = 84
                dense_input = 84
            else:
                hidden1_output = 100
                hidden2_output = 30
                dense_input = 30
            self.hidden1 = nn.Linear(embed.shape[1], hidden1_output)
            self.hidden2 = nn.Linear(hidden1_output, hidden2_output)

        self.dropout = nn.Dropout(args.dropout)

        self.num_layers = args.num_layers
        self.num_outputs = num_outputs
        self.embeddings = nn.Embedding.from_pretrained(embed, freeze=False)
        self.senti_vec = nn.Parameter(torch.FloatTensor(embed.shape[1], 1),
                                      requires_grad=True)

        self.dence = nn.Linear(dense_input, num_outputs)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

        nn.init.xavier_normal_(self.senti_vec)

    def forward(self, inputs):
        embeddings = self.embeddings(inputs)
        senti_aware_embed = torch.add(embeddings, self.senti_vec.squeeze())
        if self.num_layers == 0:
            output = self.softmax(self.dence(senti_aware_embed))
        elif self.num_layers == 1:
            output = self.relu(self.hidden1(senti_aware_embed))
            output = self.dropout(output)
            output = self.softmax(self.dence(output))
        else:
            output = self.relu(self.hidden1(senti_aware_embed))
            output = self.dropout(output)
            output = self.relu(self.hidden2(output))
            output = self.dropout(output)
            output = self.softmax(self.dence(output))

        return output

