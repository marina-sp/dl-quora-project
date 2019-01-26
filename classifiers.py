import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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
        query = self.query_l(torch.squeeze(pooled))
        keys = self.key_l(x)
        values = self.value_l(x)

        energy = torch.matmul(torch.unsqueeze(query,1), keys.transpose(1,2)) # (batchsize, seq_len)
        energy = torch.squeeze(energy,1)
        energy /= np.sqrt(keys.shape[-1]) # scale by square root of Dkey

        attention =  torch.softmax(energy, dim=1) # TODO
        attention = attention * mask

        output = torch.sum(values * torch.unsqueeze(attention,-1), dim = 1)
        return output


class LSTMClassifier(nn.Module):
    def __init__(self,
                 emb_size,
                 rnn_size = 100,
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


    def forward(self, x, lengths):
        x = torch.autograd.Variable(x)

        packed = pack_padded_sequence(x, lengths, batch_first=True)
        h0 = torch.zeros(self.lstm.num_layers * 2, len(x), self.lstm.hidden_size, device=self.device)
        c0 = torch.zeros(self.lstm.num_layers * 2, len(x), self.lstm.hidden_size, device=self.device)
        self.lstm.flatten_parameters()
        output, _ = self.lstm(packed, (h0,c0))
        output, _ = pad_packed_sequence(output, batch_first=True)
        x = output[:,-1,:]

        x = self.dropout(x)
        x = torch.tanh(x)
        x = self.output(x)
        return torch.sigmoid(x)

class AttentiveLSTMClassifier(nn.Module):
    def __init__(self,
                 emb_size,
                 rnn_size = 100,
                 attn_size = 50,
                 device=None):
        super().__init__()
        self.lstm = nn.LSTM(input_size = emb_size,
                            hidden_size = rnn_size,
                            bidirectional = True,
                            batch_first = True)
        self.dropout = nn.Dropout(p=0.1)
        self.attention = Attention(emb_size = rnn_size*2, qsize = attn_size , vsize = attn_size)
        self.output = nn.Linear(in_features=attn_size,
                                out_features=1)
        self.name = 'attn'
        self.device = device


    def forward(self, x, lengths):
        x = torch.autograd.Variable(x)

        packed = pack_padded_sequence(x, lengths, batch_first=True)
        h0 = torch.zeros(self.lstm.num_layers * 2, len(x), self.lstm.hidden_size, device=self.device)
        c0 = torch.zeros(self.lstm.num_layers * 2, len(x), self.lstm.hidden_size, device=self.device)
        self.lstm.flatten_parameters()
        output, _ = self.lstm(packed, (h0,c0))
        output, _ = pad_packed_sequence(output, batch_first=True)
        x = output[:,:,:]
        x = self.dropout(x)

        max_len = lengths.tolist()[0]
        mask = [[1.0] * l + [0.0] * (max_len - l) for l in lengths.tolist()]
        x = self.attention(x, torch.tensor(mask, device = self.device))
        x = self.dropout(x)

        x = torch.tanh(x)
        x = self.output(x)
        return torch.sigmoid(x)
