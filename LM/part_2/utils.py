''' This file contains the functions and classes for data loading and preprocessing '''

import torch
import torch.utils.data as data


def read_file(path, eos_token="<eos>"):
    ''' This function reads a file and returns a list of sentences '''
    output = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output


class Lang():
    '''
    This class computes and stores our vocabulary and implements Word to ids and ids to word methods
    '''
    def __init__(self, corpus, special_tokens=None):
        if special_tokens is None:
            special_tokens = []
        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v:k for k, v in self.word2id.items()}

    def get_vocab(self, corpus, special_tokens=None):
        ''' This function computes the vocabulary of a corpus '''
        if special_tokens is None:
            special_tokens = []
        output = {}
        i = 0
        for st in special_tokens:
            output[st] = i
            i += 1
        for sentence in corpus:
            for w in sentence.split():
                if w not in output:
                    output[w] = i
                    i += 1
        return output


class PennTreeBank (data.Dataset):
    '''
    This class takes as input a corpus and class lang creating a vocabulary with an id for each word
    The class store two lists with the source sequences and target sequences of words
    '''

    def __init__(self, corpus, lang):
        self.source = []
        self.target = []

        for sentence in corpus:
            self.source.append(sentence.split()[0:-1])  # From the first token till the second-last
            self.target.append(sentence.split()[1:])    # From the second token till the last token

        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        src= torch.LongTensor(self.source_ids[idx])
        trg = torch.LongTensor(self.target_ids[idx])
        sample = {'source': src, 'target': trg}
        return sample

    # Auxiliary methods

    def mapping_seq(self, sequences, lang):
        ''' Map sequences of tokens to corresponding computed in Lang class '''
        res = []
        for seq in sequences:
            tmp_seq = []
            for x in seq:
                if x in lang.word2id:
                    tmp_seq.append(lang.word2id[x])
                else:
                    print('OOV found!')
                    print('You have to deal with that') # Note: PennTreeBank doesn't have OOV
                    break
            res.append(tmp_seq)
        return res


def collate_fn(batch, pad_token):
    '''
    This function takes as input a batch of items from a dataset,
    the batch contains pairs of source and target sequences of various lengths,
    the function sorts them according to their length and stacks them 
    in a matrix with dimension batch * max_len.
    The pad_token is used to obtain sequences with the same length
    '''

    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths

    # Sort data by seq lengths

    batch.sort(key=lambda x: len(x["source"]), reverse=True)
    new_item = {}
    for key in batch[0].keys():
        new_item[key] = [d[key] for d in batch]

    source, _ = merge(new_item["source"])
    target, lengths = merge(new_item["target"])

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    new_item["source"] = source.to(device)
    new_item["target"] = target.to(device)
    new_item["number_tokens"] = sum(lengths)
    return new_item
