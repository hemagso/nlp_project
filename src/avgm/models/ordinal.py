import torch.nn as nn
import torch
from ..train.modules import CumulativeLogisticLink


class AVGMOrdinal(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, pad_index):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_index)
        self.rnn = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2 * hidden_size, 1)
        self.logit_link = CumulativeLogisticLink(output_size)

    def forward(self, tokens, lengths):
        embedded = self.embedding(tokens)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        hidden = torch.cat((hidden[0, :, :], hidden[1, :, :]), dim=1)
        linear = self.fc(hidden)
        logit = self.logit_link(linear)
        return logit