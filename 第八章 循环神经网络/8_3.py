import random
import torch
from d2l import torch as d2l
from matplotlib import pyplot as plt
import numpy as np

tokens = d2l.tokenize(d2l.read_time_machine())
corpus = [token for line in tokens for token in line]
vocab = d2l.Vocab(corpus)
freqs = [freq for token, freq in vocab.token_freqs]

bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
bigram_vocab = d2l.Vocab(bigram_tokens)
bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]

trigram_tokens = [triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = d2l.Vocab(trigram_tokens)
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]

length = len(freqs)
x = range(1, length+1)
bigram_length = len(bigram_freqs)
biagram_x = range(1, bigram_length + 1)
trigram_length = len(trigram_freqs)
trigram_x = range(1, trigram_length + 1)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
# 绘制第一个子图
ax1.plot(freqs, x, label='token: x', color='blue', linestyle='-', marker='o', markersize=1)
ax1.set_title("One word")
ax1.legend()

# 绘制第二个子图
ax2.plot(bigram_freqs, biagram_x, label='token: x', color='red', linestyle='-', marker='o', markersize=1)
ax2.set_title("Two word")
ax2.legend()

ax3.plot(trigram_freqs, trigram_x, label='token: x', color='purple', linestyle='-', marker='o', markersize=1)
ax3.set_title("Three word")
ax3.legend()
plt.tight_layout()  # 调整子图布局以防止重叠
# 显示图表
plt.show()

def seq_data_iter_random(corpus, batch_size, num_steps):
    corpus = corpus[random.randint(0, num_steps - 1):]
    num_subseqs = (len(corpus) - 1) // num_steps
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    random.shuffle(initial_indices)

    def data(pos):
        return corpus[pos: pos + num_steps]
    
    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)

def seq_data_iter_squential(corpus, batch_size, num_steps):
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset -1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i+num_steps]
        Y = Ys[:, i: i+num_steps]
        yield X, Y

class SeqDataLoader:
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = d2l.seq_data_iter_random
        else:
            self.data_iter_fn = d2l.seq_data_iter_sequential
        self.corpus, self.vocab = d2l.load_time_machine(max_tokens)
        self.batch_size = batch_size
        self.num_steps = num_steps
    
    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)
    

def load_data_time_machine(batch_size, num_steps, use_random_iter = False, max_tokens = -1):
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab