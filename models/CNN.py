"""
CNN
:author: Qizhi Li
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self, embed, num_outputs, max_seq_length, num_filters=100,
                 is_freeze=False):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.filter_sizes = (3, 4, 5)
        self.embeddings = nn.Embedding.from_pretrained(embed, freeze=is_freeze)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (k, embed.shape[1])) for k in self.filter_sizes]
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(num_filters * len(self.filter_sizes), num_outputs)
        self.softmax = nn.Softmax()

    def pool(self, x):
        """
        Global max pooling layer
        :param x: tensor
                shape: (batch_size, output, height)
        :return: tensor
                shape: (batch_size, output, 1)
        """
        return F.max_pool1d(x, kernel_size=x.shape[2])

    def forward(self, inputs):
        """
        :param inputs: tensor
                shape: (batch_size, seq_len)
        :return out: tensor
                shape: (batch_size, num_outputs)
        """
        # height_1: seq_len, width_1: w2v_dim
        embed = self.embeddings(inputs)  # shape: (batch_size, height_1, width_1)
        # Adding a dimension, shape: (batch_size, in_channel, height_1, width_1)
        embed = embed.unsqueeze(1)

        pool_outs = []
        i = 0
        for conv in self.convs:
            # height_2: height_1 - filter_size + 1, width_2 = 1
            # shape: (batch_size, output, height_2, width_2)
            conv_out = conv(embed)
            conv_relu_out = self.relu(conv_out).squeeze(3)  # clean up width
            # shape: (batch_size, out, 1)
            pool_out = self.pool(conv_relu_out).squeeze(2)  # clean up height
            pool_outs.append(pool_out)
            i += 1

        # shape: (batch_size, out * len(filter_sizes))
        pool_outs = torch.cat(pool_outs, 1)
        pool_dropout_out = self.dropout(pool_outs)
        out = self.linear(pool_dropout_out)
        return self.softmax(out)