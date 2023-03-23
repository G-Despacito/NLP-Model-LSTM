import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence

class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_class, p_dropout):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.enc = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim // 2, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=p_dropout if num_layers > 1 else 0.)
        
        self.drop = nn.Dropout(p_dropout)
        self.classifier = nn.Linear(hidden_dim, num_class)

    def forward(self, x):
        x_len = torch.sum(x!=0, dim=-1)
        x_embed = self.embedding(x)

        # LSTM
        origin_len = x_embed.shape[1]
        lengths, sorted_idx = x_len.sort(0, descending=True)
        x_embed = x_embed[sorted_idx]
        inp = pack_padded_sequence(x_embed, lengths, batch_first=True)
        out, _ = self.enc(inp)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=origin_len)
        _, unsorted_idx = sorted_idx.sort(0)
        out = out[unsorted_idx]
        out = self.drop(out)
        
        # classifier
        logits = self.classifier(out[:, 0, :])
        logits = F.log_softmax(logits, dim=-1)

        return logits
