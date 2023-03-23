import torch
import nltk
import csv

from collections import Counter
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, source_file, max_len, min_occurance, word2idx=None):
        data, word_counter = self.extract_data_from_source(source_file)

        if word2idx == None:
            self.word2idx = self.build_word2idx(word_counter, min_occurance)
        else:
            self.word2idx = word2idx

        self.idxs, self.sents, self.labels = self.convert_data(data, self.word2idx, max_len)

    def extract_data_from_source(self, source):
        word_counter = Counter()
        data = []

        with open(source, 'r') as f:
            reader = csv.reader(f)
            next(reader) # skip the first line

            for line in reader:
                idx, sent, label = int(line[0]), line[1], int(line[2])

                tokens = nltk.word_tokenize(sent)
                word_counter.update(tokens)
                data.append((idx, tokens, label))
        
        return data, word_counter
    
    def build_word2idx(self, word_counter, min_occurance):
        word2idx = {}
        word2idx['[PAD]'] = 0
        word2idx['[UNK]'] = 1

        for _, word in enumerate(word_counter, 2):
            if word_counter[word] > min_occurance:
                word2idx[word] = len(word2idx)
        
        return word2idx

    def convert_data(self, data, word2idx, max_len):
        idxs, sents, labels = [], [], []
        for line in data:
            idx, sent, label = line[0], line[1], line[2]

            if len(sent) < max_len:
                sent += ['[PAD]' for _ in range(max_len - len(sent))]
            else:
                sent = sent[:max_len]

            tokens = [word2idx.get(word, word2idx['[UNK]']) for word in sent]
            
            idxs.append(idx)
            sents.append(tokens)
            labels.append(label)

        idxs, sents, labels = torch.LongTensor(idxs), torch.LongTensor(sents), torch.LongTensor(labels)

        return idxs, sents, labels
    
    def __getitem__(self, index):
        return (self.idxs[index], self.sents[index], self.labels[index])

    def __len__(self):
        return self.sents.shape[0]
