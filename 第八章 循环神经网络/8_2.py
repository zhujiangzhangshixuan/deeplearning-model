import collections
import re
from d2l import torch as d2l

d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt','090b5e7e70c295757f55df93cb0a180b9691891a')
def read_time_machine():
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]




def tokenize(lines, token='word'):
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]

class Vocab:    
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        counter  = collections.Counter([token for line in tokens for token in line])
        self.idx_to_tokens = ['unk'] + reserved_tokens
        self.tokens_to_idx = {token : idx for idx, token in enumerate(self.idx_to_tokens)}
        self._token_freqs = sorted(counter.items(), key = lambda x: x[1], reverse = True)
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.tokens_to_idx:
                self.idx_to_tokens.append(token)
                self.tokens_to_idx[token] = len(self.idx_to_tokens) - 1
    
    def __len__(self):
        return len(self.idx_to_tokens)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (tuple, list)):
            return self.tokens_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
            
    def count_corpus(tokens):
        if len(tokens) == 0 or isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        return collections.Counter(tokens)
    
    def unk(self):
        return 0

    def token_freqs(self):
        return self.token_freqs


def load_corpus_time_machine(max_tokens=-1):
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab  = Vocab(tokens)
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

lines = read_time_machine()
tokens = tokenize(lines, token='char')

"""
tokens = tokenize(lines)
bigram = [pair for pair in zip(tokens[:-1], tokens[1:])]
bivocab = Vocab(bigram)
print(bivocab._token_freqs[0])
"""


