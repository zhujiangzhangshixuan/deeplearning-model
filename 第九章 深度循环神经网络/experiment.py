import pandas as pd
import re
import collections
import random
import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F
import math
from tqdm import tqdm
import os

txt_folder = 'D:\动手学深度学习pytorch\深度循环神经网络\data'
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
vocab = Vocab(tokens)
tokens = [token for line in tokens for token in line]
corpus = vocab[tokens]

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

batch_size = 32
num_steps = 35
class SeqDataLoader:
    def __init__(self, batch_size, num_steps, corpus):
        self.data_iter_fn = seq_data_iter_sequential
        self.corpus = corpus
        self.batch_size, self.num_steps = batch_size, num_steps
    
    def __iter__(self):
        return self.data_iter_fn(corpus=corpus, batch_size=batch_size, num_steps=num_steps)
train_iter = SeqDataLoader(batch_size, num_steps, corpus)


#RNN模型架构
class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_direntions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)
    
    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        output, state = self.rnn(X, state)
        Y = self.linear(output.reshape((-1, output.shape[-1])))
        return Y, state
    
    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            return torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device)
        else:
            return (torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device), torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device))

#RNN模型测试
def predict_rnn(prefix, num_preds, net, vocab, device):
    state = net.begin_state(device)
    outputs = [vocab[prefix[0]]]
    get_inputs = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:
        _, state = net(get_inputs(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):
        y, state = net(get_inputs(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])

#RNN模型训练的tricky
def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad*p.grad)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


#RNN模型训练
def train_epoch(net, train_iter, loss, updater, device, use_random_iter, total_batches):
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)
    for batch_idx, (X, y) in enumerate(tqdm(train_iter, total=total_batches, desc=f"Training process")):
        if state is None or use_random_iter:
            state = net.begin_state(device=device, batch_size=X.shape[0])
        else:
            if not isinstance(state, tuple):
                state.detach_()
            else:
                for s in state:
                    s.detach_()
        X, y = X.to(device), y.to(device)
        y = y.T.reshape(-1)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        updater.zero_grad()
        l.backward()
        
        grad_clipping(net, 1)
        
        updater.step()
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0]/metric[1]), metric[1]/timer.stop()

def train(net, train_iter, loss, updater, device, use_random_iter, vocab, num_epochs, save_path='rnn_model.pth'):
    predict = lambda prefix: predict_rnn(prefix, num_preds=30, net=net, vocab=vocab, device=device)
    total_batches = 0
    for X, y in train_iter:
        total_batches = total_batches + 1
    print(f"Total batchs: {total_batches}")
    for epoch in range(num_epochs):
        ppl, speed = train_epoch(net, train_iter, loss, updater, device, use_random_iter, total_batches)
        if epoch % 1 == 0:
            print(f"Epoch:{epoch}")
            print(predict('time traveller'))
            print(f"困惑度：{ppl:.2f}")
            torch.save(net.state_dict(), save_path)
    print(f"困惑度{ppl:.1f},speed:{speed:.1f}词元/秒{str(device)}")
    print(predict("time traveller"))
    print(predict("hello"))
    


device = d2l.try_gpu()  
vocab_size = len(vocab)
num_hiddens = 256
num_inputs = vocab_size
rnn_layer = nn.LSTM(num_inputs, num_hiddens, num_layers=2)
net = RNNModel(rnn_layer, vocab_size)
loss = nn.CrossEntropyLoss()
updater = torch.optim.SGD(net.parameters(), lr=0.5)
num_epochs = 500
use_random_iter = False

net.load_state_dict(torch.load('rnn_model.pth'))

print("Training the model")
train(net, train_iter, loss, updater, device, use_random_iter, vocab, num_epochs)

"""
predict = predict_rnn("fuck you", 100, net, vocab, device)
print(predict)
"""
