''' This file contains the architecture for the language model '''


import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ModelIAS(nn.Module):
    ''' Model for Intent and Slot Filling '''
    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0, bidirectional=False, dropout_flag=False):
        super(ModelIAS, self).__init__()

        self.bidirectional = bidirectional
        self.dropout_flag = dropout_flag

        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)

        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=bidirectional, batch_first=True)
        if self.bidirectional:
            self.slot_out = nn.Linear(hid_size*2, out_slot)
            self.intent_out = nn.Linear(hid_size*2, out_int)
        else:
            self.slot_out = nn.Linear(hid_size, out_slot)
            self.intent_out = nn.Linear(hid_size, out_int)

        if self.dropout_flag:
            self.dropout = nn.Dropout(0.1)

    def forward(self, utterance, seq_lengths):
        ''' Forward pass of the model '''
        # utterance.size() = batch_size X seq_len
        utt_emb = self.embedding(utterance) # utt_emb.size() = batch_size X seq_len X emb_size
        if self.dropout_flag:
            utt_emb = self.dropout(utt_emb)
        # pack_padded_sequence avoid computation over pad tokens reducing the computational cost

        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        # Process the batch
        packed_output, (last_hidden, _ ) = self.utt_encoder(packed_input)

        # Unpack the sequence
        utt_encoded, _ = pad_packed_sequence(packed_output, batch_first=True)
        # Get the last hidden state
        if self.bidirectional:
            last_hidden = torch.cat((last_hidden[-2], last_hidden[-1]), dim=1)
        else:
            last_hidden = last_hidden[-1,:,:]

        # Compute slot logits
        if self.dropout_flag:
            utt_encoded = self.dropout(utt_encoded)
        slots = self.slot_out(utt_encoded)
        # Compute intent logits
        if self.dropout_flag:
            last_hidden = self.dropout(last_hidden)
        intent = self.intent_out(last_hidden)

        # Slot size: batch_size, seq_len, classes
        slots = slots.permute(0,2,1) # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        return slots, intent


def init_weights(mat):
    ''' Initialize weights for the model '''
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
                if m.bias != None:
                    m.bias.data.fill_(0.01)
