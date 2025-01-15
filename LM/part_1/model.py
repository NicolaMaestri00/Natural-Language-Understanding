''' This file contains the architecture for the language model '''


import torch
import torch.nn as nn


class LmRnn(nn.Module):
    ''' Language Model based on a Vanilla RNN   '''

    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, n_layers=1):
        super(LmRnn, self).__init__()

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.rnn = nn.RNN(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        ''' Forward pass '''
        emb = self.embedding(input_sequence)
        rnn_out, _  = self.rnn(emb)
        output = self.output(rnn_out).permute(0,2,1)
        return output


class LmLstm(nn.Module):
    ''' Language Model based on a LSTM '''

    def __init__(self,
                 emb_size,
                 hidden_size,
                 output_size,
                 pad_index=0,
                 dropout=False,
                 out_dropout=0.1,
                 emb_dropout=0.1,
                 n_layers=1):
        super(LmLstm, self).__init__()

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        self.output = nn.Linear(hidden_size, output_size)

        # Dropout Layers
        self.dropout = dropout
        if self.dropout:
            self.dropout_emb = nn.Dropout(emb_dropout)
            self.dropout_out = nn.Dropout(out_dropout)

    def forward(self, input_sequence):
        ''' Forward pass '''
        emb = self.embedding(input_sequence)
        if self.dropout:
            emb = self.dropout_emb(emb)
        lstm_out, _  = self.lstm(emb)
        if self.dropout:
            lstm_out = self.dropout_out(lstm_out)
        output = self.output(lstm_out).permute(0,2,1)
        return output


def init_weights(mat):
    ''' Initialize the weights of the model '''
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
