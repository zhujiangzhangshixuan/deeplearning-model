import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

T = 1000
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))


tau = 4
features = torch.zeros((T-tau, tau))
for i in range(tau):
    features[:, i] = x[i: T - tau + i]
labels = x[tau:].reshape((-1, 1))
batch_size, n_train = 16, 600
train_iter = d2l.load_array((features[:n_train], labels[:n_train]), batch_size, is_train=True)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def get_net(tau):
    net = nn.Sequential(nn.Linear(tau, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1))
    net.apply(init_weights)
    return net
loss = nn.MSELoss()

def train(net, train_iter, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
        print(f'epoch {epoch}, loss {d2l.evaluate_loss(net, train_iter, loss):f}')
net = get_net(tau)
train(net, train_iter, 5, 0.001)
onestep_preds = net(features)
time = time.detach().numpy()
onestep_preds = onestep_preds.detach().numpy()
multistep_preds = torch.zeros(T)
multistep_preds[:n_train + tau] = x[:n_train + tau]
for i in range(n_train + tau, T):
    multistep_preds[i] = net(multistep_preds[i-tau: i]).reshape((1, -1))
multistep_preds = multistep_preds.detach().numpy()

fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
# 绘制第一个子图
ax1.plot(time[tau:], x[tau:], label='Line 1', color='blue', linestyle='-', marker='o', markersize=1)
ax1.set_title("data")
ax1.legend()

# 绘制第二个子图
ax2.plot(time[tau:], onestep_preds, label='Line 2', color='red', linestyle='--', marker='s', markersize=1)
ax2.set_title("One-steps preds")
ax2.legend()

ax3.plot(time[tau:], multistep_preds[tau:], label='Line 3', color='purple', linestyle='--', marker='s', markersize=1)
ax3.set_title("Multi-steps preds")
ax3.legend()
plt.tight_layout()  # 调整子图布局以防止重叠
# 显示图表
plt.show()



