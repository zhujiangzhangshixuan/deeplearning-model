import os
import re
import collections
import torch
import random

def get_data_tokens(txt_folder):
    tokens = []
    for filename in os.listdir(txt_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(txt_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                lines = [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]
                token = [list(line) for line in lines]
                for token_single in token:
                    tokens.append(token_single)
    return tokens

class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        counter = self.count_corpus(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self.token_freqs:
            if freq < min_freq:
                break
            if token not in self.idx_to_token:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
    
    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk())
        return [self.__getitem__(token) for token in tokens]
    
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.to_tokens(indice) for indice in indices]

    def unk(self):
        return 0
    
    def token_freqs(self):
        return self.token_freqs
    
    def count_corpus(self, tokens):
        if len(tokens) == 0 or isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        return collections.Counter(tokens)
    

def seq_data_iter_sequential(corpus, batch_size, num_steps):
    offset = random.randint(0, num_steps)   
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batchs = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batchs, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y
    
class SeqDataLoader:
    def __init__(self, batch_size, num_steps, corpus):
        self.data_iter_fn = seq_data_iter_sequential
        self.corpus = corpus
        self.batch_size, self.num_steps = batch_size, num_steps
    
    def __iter__(self):
        return self.data_iter_fn(corpus=self.corpus, batch_size=self.batch_size, num_steps=self.num_steps)