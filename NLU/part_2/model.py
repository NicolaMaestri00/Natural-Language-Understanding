''' This file contains the architecture for the language model '''


import torch.nn as nn
from transformers import BertModel


class Bert_slot_intent_classifier(nn.Module):
    ''' Bert_intent_classifier
    '''

    def __init__(self, pretrained_model_name="bert-base-uncased", num_intents=26, num_slots=130, dropout=0.1):
        super(Bert_slot_intent_classifier, self).__init__()

        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.dropout_layer = nn.Dropout(dropout)
        self.intent_classifier = nn.Linear(self.bert.config.hidden_size, num_intents)
        self.slot_classifier = nn.Linear(self.bert.config.hidden_size, num_slots)

    def forward(self, x):
        ''' Forward pass '''
        x = self.bert(x)
        CLS_emb = x.pooler_output
        CLS_emb = self.dropout_layer(CLS_emb)
        intent_logits = self.intent_classifier(CLS_emb)

        last_hidden_state = x.last_hidden_state
        last_hidden_state = self.dropout_layer(last_hidden_state)
        slot_logits = self.slot_classifier(last_hidden_state)
        slot_logits = slot_logits.permute(0, 2, 1)

        return intent_logits, slot_logits
