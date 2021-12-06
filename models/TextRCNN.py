# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextRCNN(nn.Module):
    def __init__(self, embed, num_hiddens, bidirectional, num_outputs, max_seq_length):
        super().__init__()
        hidden_size = (num_hiddens * 2) if bidirectional else num_hiddens
        self.embedding = nn.Embedding.from_pretrained(embed, freeze=False)
        self.lstm = nn.LSTM(embed.shape[1], num_hiddens, 1,
                            bidirectional=bidirectional, batch_first=True, dropout=0.5)
        self.maxpool = nn.MaxPool1d(max_seq_length)
        self.fc = nn.Linear(hidden_size + embed.shape[1], num_outputs)

    def forward(self, inputs):
        embed = self.embedding(inputs)
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        out = self.fc(out)
        return out
