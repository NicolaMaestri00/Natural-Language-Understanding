''' This file contains the architecture for the language model '''


import torch
import torch.nn as nn


class VariationalDropout(nn.Module):
    ''' Variational Dropout Layer '''
    def __init__(self, p: float):
        """
        Args:
            p (float): The dropout probability.
        """
        super(VariationalDropout, self).__init__()
        self.p = p

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, sequence_length, hidden_size).

        Returns:
            torch.Tensor: Output tensor with the same shape as x, but with consistent dropout mask.
        """
        if not self.training or self.p == 0.0:
            # If not in training mode or p=0, return input as is
            return x

        # Create a dropout mask for the hidden dimension only
        dropout_mask = (torch.rand(x.size(0), 1, x.size(2), device=x.device) > self.p).float()

        # Scale by (1 / (1 - p)) to maintain the expected value of activations
        dropout_mask /= (1 - self.p)

        # Apply the mask to the input, keeping it consistent across the sequence length dimension
        return x * dropout_mask


class LmLstmVD(nn.Module):
    '''
    Model Architecture
    '''
    def __init__(self,
                 emb_size,
                 hidden_size,
                 output_size,
                 pad_index=0,
                 out_dropout=0.1,
                 emb_dropout=0.1,
                 n_layers=1
                 ):

        super(LmLstmVD, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        self.output = nn.Linear(hidden_size, output_size)

        # Dropout Layers
        self.variational_dropout_emb = VariationalDropout(emb_dropout)
        self.variational_dropout_out = VariationalDropout(out_dropout)

    def forward(self, input_sequence):
        ''' Forward pass '''
        emb = self.embedding(input_sequence)
        emb = self.variational_dropout_emb(emb)
        lstm_out, _  = self.lstm(emb)
        lstm_out = self.variational_dropout_out(lstm_out)
        output = self.output(lstm_out).permute(0,2,1)
        return output


class LmLstmWT(nn.Module):
    '''
    Model Architecture
    '''
    def __init__(self,
                 emb_size,
                 hidden_size,
                 output_size,
                 pad_index=0,
                 out_dropout=0.1,
                 emb_dropout=0.1,
                 n_layers=1
                 ):

        super(LmLstmWT, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.hidden_to_emb = nn.Linear(hidden_size, emb_size)
        self.pad_token = pad_index

        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(emb_size, output_size)
        self.output.weight = self.embedding.weight

        # Dropout Layers
        self.dropout_emb = nn.Dropout(emb_dropout)
        self.dropout_out = nn.Dropout(out_dropout)

    def forward(self, input_sequence):
        ''' Forward pass '''
        emb = self.embedding(input_sequence)
        emb = self.dropout_emb(emb)
        lstm_out, _  = self.lstm(emb)
        lstm_out = self.hidden_to_emb(lstm_out)
        lstm_out = self.dropout_out(lstm_out)
        output = self.output(lstm_out).permute(0,2,1)
        return output


class LmLstm(nn.Module):
    '''
    Model Architecture
    '''
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
        # Token ids to vectors, we will better see this in the next lab
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # Pytorch's RNN layer: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space
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


class LmLstmWTVD(nn.Module):
    '''
    Model Architecture
    '''
    def __init__(self,
                 emb_size,
                 hidden_size,
                 output_size,
                 pad_index=0,
                 out_dropout=0.1,
                 emb_dropout=0.1,
                 n_layers=1):

        super(LmLstmWTVD, self).__init__()
        # Token ids to vectors, we will better see this in the next lab
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # Pytorch's RNN layer: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.hidden_to_emb = nn.Linear(hidden_size, emb_size)
        self.pad_token = pad_index

        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(emb_size, output_size)
        self.output.weight = self.embedding.weight

        # Dropout Layers
        self.variational_dropout_emb = VariationalDropout(emb_dropout)
        self.variational_dropout_out = VariationalDropout(out_dropout)


    def forward(self, input_sequence):
        ''' Forward pass '''
        emb = self.embedding(input_sequence)
        emb = self.variational_dropout_emb(emb)
        lstm_out, _  = self.lstm(emb)
        lstm_out = self.hidden_to_emb(lstm_out)
        lstm_out = self.variational_dropout_out(lstm_out)
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
