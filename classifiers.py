import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class Attention(nn.Module):
    def __init__(self, emb_size, qsize = 50, vsize = 100):
        super().__init__()
        self.query_l = nn.Linear(emb_size, qsize)
        self.key_l = nn.Linear(emb_size, qsize)
        self.value_l = nn.Linear(emb_size, vsize)

    def distribute(self, seq, layer):
        return torch.stack([layer(elem) for elem in seq])

    def forward(self, x, mask):
        # x shape: batch_size x seq_len x emb_size

        pooled = F.avg_pool2d(x,(x.shape[1],1))
        squeezed = torch.squeeze(pooled)
        query = self.query_l(squeezed)
        keys = self.key_l(x)
        values = self.value_l(x)

        energy = torch.matmul(torch.unsqueeze(query,1), keys.transpose(1,2)) # (batchsize, seq_len)
        energy = torch.squeeze(energy,1)
        energy /= np.sqrt(keys.shape[-1]) # scale by square root of Dkey

        attention =  torch.softmax(energy, dim=1) # TODO
        attention = attention * mask

        output = torch.sum(values * torch.unsqueeze(attention,-1), dim = 1)
        return output

class AttentionClassifier(nn.Module):
    def __init__(self, emb_size, hidden_size = 100):
        super().__init__()
        self.attention = Attention(emb_size = emb_size, vsize = hidden_size)
        self.dropout = nn.Dropout(p=0.1)
        self.output = nn.Linear(in_features= hidden_size,
                                out_features=1)
        self.name = 'attn'

    def forward(self, x, attention_mask = None):
        x = torch.autograd.Variable(x)
        x = self.attention(x, attention_mask)
        x = self.dropout(x)
        x = torch.tanh(x)
        x = self.output(x)
        return torch.sigmoid(x)


class LSTMClassifier(nn.Module):
    def __init__(self,
                 emb_size,
                 rnn_size = 30,
                 device=None):
        super().__init__()
        self.lstm = nn.LSTM(input_size = emb_size,
                            hidden_size = rnn_size,
                            bidirectional = True,
                            batch_first = True)
        self.dropout = nn.Dropout(p=0.1)
        self.output = nn.Linear(in_features=rnn_size*2,
                                out_features=1)
        self.name = 'lstm'
        self.device = device


    def forward(self, x, batch_size, lengths):
        x = torch.autograd.Variable(x)
        lengths.sort(reverse=True)

        packed = pack_padded_sequence(x, lengths, batch_first=True)
        h0 = torch.zeros(self.lstm.num_layers * 2, batch_size, self.lstm.hidden_size, device=self.device)
        c0 = torch.zeros(self.lstm.num_layers * 2, batch_size, self.lstm.hidden_size, device=self.device)
        output, _ = self.lstm(packed, (h0,c0))
        output, _ = pad_packed_sequence(output, batch_first=True)
        x = output[:,-1,:]

        x = self.dropout(x)
        x = torch.tanh(x)
        x = self.output(x)
        return torch.sigmoid(x)
