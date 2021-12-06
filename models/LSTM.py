"""
Author: Qizhi Li
LSTM或BiLSTM模型
"""
import torch
import torch.nn as nn


class LSTM(nn.Module):

    def __init__(self, embed, num_inputs, num_hiddens, num_outputs, bidirectional=False,
                 num_layers=1, is_freeze=False):
        super().__init__()
        self.linear_hiddens = (2 * num_hiddens) if bidirectional else num_hiddens

        self.embeddings = nn.Embedding.from_pretrained(embed, freeze=is_freeze)
        self.lstm_layer = nn.LSTM(input_size=num_inputs, hidden_size=num_hiddens,
                                  batch_first=True, bidirectional=bidirectional,
                                  num_layers=num_layers, dropout=0.5)
        self.linear = nn.Linear(self.linear_hiddens, num_outputs)
        self.softmax = nn.Softmax()

    def forward(self, inputs):
        """
        :param inputs: tensor
                shape: (batch_size, seq_length, w2v_dim)
        :return y_hat: tensor
                shape: (D * num_layers, batch_size, num_hiddens)
        """
        embed = self.embeddings(inputs)
        rnn_output, _ = self.lstm_layer(embed)
        rnn_last_output = rnn_output[:, -1, :]
        linear_output = self.linear(rnn_last_output)
        y_hat = self.softmax(linear_output)

        return y_hat
