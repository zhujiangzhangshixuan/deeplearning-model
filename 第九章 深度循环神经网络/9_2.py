import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F
import math

batch_size, num_steps = 32, 35

train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)


def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size
    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01
    
    def three():
        return (normal((num_inputs, num_hiddens)), normal((num_hiddens, num_hiddens)), torch.zeros(num_hiddens, device=device))
    
    W_xi, W_hi, b_i = three()
    W_xf, W_hf, b_f = three()
    W_xo, W_ho, b_o = three()
    W_xc, W_hc, b_c = three()
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), torch.zeros((batch_size, num_hiddens), device=device))

def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
        O = torch.sigmoid((X @ W_xo + H @ W_ho) + b_o)
        C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * torch.tanh(C)
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, C)


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
def train_epoch(net, train_iter, loss, updater, device, use_random_iter):
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)
    for X, Y in train_iter:
        if state is None or use_random_iter:
            state = net.begin_state(device=device, batch_size=X.shape[0])
        else:
            if not isinstance(state, tuple):
                state.detach_()
            else:
                for s in state:
                    s.detach_()
        X, y = X.to(device), Y.to(device)
        y = y.T.reshape(-1)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        updater.zero_grad()
        l.backward()
        """
        grad_clipping(net, 1)
        """
        
        updater.step()
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0]/metric[1]), metric[1]/timer.stop()

def train(net, train_iter, loss, updater, device, use_random_iter, vocab, num_epochs):
    predict = lambda prefix: predict_rnn(prefix, num_preds=30, net=net, vocab=vocab, device=device)
    for epoch in range(num_epochs):
        ppl, speed = train_epoch(net, train_iter, loss, updater, device, use_random_iter)
        if epoch % 50 == 0:
            print(f"Epoch:{epoch}")
            print(predict('time traveller'))
            print(f"困惑度：{ppl:.2f}")
    print(f"困惑度{ppl:.1f},speed:{speed:.1f}词元/秒{str(device)}")
    print(predict("time traveller"))
    print(predict("hello"))

device = d2l.try_gpu()  
vocab_size = len(vocab)
num_hiddens = 256
num_inputs = vocab_size
rnn_layer = nn.GRU(num_inputs, num_hiddens)
net = RNNModel(rnn_layer, vocab_size)
loss = nn.CrossEntropyLoss()
updater = torch.optim.SGD(net.parameters(), lr=1)
num_epochs = 800
use_random_iter = False
train(net, train_iter, loss, updater, device, use_random_iter, vocab, num_epochs)
    














