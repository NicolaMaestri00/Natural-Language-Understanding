''' This file contains the functions and classes for data loading and preprocessing '''


import json
from sklearn.model_selection import train_test_split
from collections import Counter
from transformers import BertTokenizer
import torch
import torch.utils.data as data


def load_data(path):
    '''
        input: path/to/data
        output: json
    '''
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset


def get_data(train_data_path, test_data_path, val_proportion=0.10):
    '''
        input: train_data_path, test_data_path, val_proportion
        output: train_data, test_data, val_data
    '''
    # Temporary Training and Test data
    tmp_train_raw = load_data(train_data_path)
    test_raw = load_data(test_data_path)

    intents = [x['intent'] for x in tmp_train_raw] # We stratify on intents
    count_y = Counter(intents)

    labels = []
    inputs = []
    mini_train = []

    for id_y, y in enumerate(intents):
        if count_y[y] > 1: # If some intents occurs only once, we put them in training
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])

    # Random Stratify
    x_train, x_dev, _, _ = train_test_split(inputs,
                                            labels,
                                            test_size=val_proportion,
                                            random_state=42,
                                            shuffle=True,
                                            stratify=labels)
    x_train.extend(mini_train)

    # Training and Validation data
    train_raw = x_train
    dev_raw = x_dev

    return train_raw, test_raw, dev_raw


class Lang():
    ''' Class to map words and labels to numbers '''
    def __init__(self, intents, slots, pad_token=0):
        self.pad_token = pad_token
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}

    def w2id(self, elements, cutoff=None, unk=True):
        ''' Map words to numbers '''
        vocab = {'pad': self.pad_token}
        if unk:
            vocab['unk'] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab

    def lab2id(self, elements, pad=True):
        ''' Map labels to numbers '''
        vocab = {}
        if pad:
            vocab['pad'] = self.pad_token
        for elem in elements:
            vocab[elem] = len(vocab)
        return vocab


class IntentsAndSlots (data.Dataset):
    ''' Dataset class for intents and slots '''

    def __init__(self, dataset, lang, unk='unk', tokenizer_name='bert-base-uncased'):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.utt_ids = self.tokenizer(self.utterances, return_tensors="pt", padding=True)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        # Get utterance
        utt = self.utt_ids['input_ids'][idx]
        utt = utt[utt.nonzero().squeeze().detach()]

        # Get intent
        intent = self.intent_ids[idx]

        # Get Slots
        mask = [0]
        for word in self.utterances[idx].split():
            mask.extend([1]+[0] * (len(self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(word))) - 3))
        mask.append(0)
        mask = torch.Tensor(mask)

        slots = torch.zeros_like(torch.Tensor(mask))
        t = 0
        for i, el in enumerate(mask):
            if el == 1:
                slots[i] = self.slot_ids[idx][t]
                t += 1


        sample_ids = {'utterance': utt, 'slots': slots, 'intent': intent, 'mask': mask}
        return sample_ids

    # Auxiliary methods

    def mapping_lab(self, sequences, mapper):
        ''' Map labels to numbers '''
        return [mapper[x] if x in mapper else mapper[self.unk] for x in sequences]

    def mapping_seq(self, sequences, mapper):
        ''' Map sequences to numbers '''
        res = []
        for seq in sequences:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            res.append(tmp_seq)
        return res


def collate_fn(batch, pad_token=0):
    '''
        input: batch, pad_token
        output: padded batch
    '''
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq
        padded_seqs = padded_seqs.detach()
        return padded_seqs, lengths

    # Sort data by seq lengths
    batch.sort(key=lambda x: len(x['utterance']), reverse=True)
    new_item = {}
    for key in batch[0].keys():
        new_item[key] = [d[key] for d in batch]

    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(new_item['utterance'])
    y_slots, y_lengths = merge(new_item["slots"])
    masks, _ = merge(new_item["mask"])
    intent = torch.LongTensor(new_item["intent"])

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    src_utt = src_utt.to(device)
    y_slots = y_slots.to(device)
    masks = masks.to(device)
    intent = intent.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)

    new_item["utterances"] = src_utt
    new_item["y_slots"] = y_slots
    new_item["masks"] = masks
    new_item["intents"] = intent
    new_item["slots_len"] = y_lengths

    return new_item
