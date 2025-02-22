import torch
import torch.nn as nn

# 定义GRU
gru = nn.GRU(input_size=10, hidden_size=20, num_layers=2)

# 输入数据
X = torch.randn(5, 3, 10)  # (seq_len, batch_size, input_size)

# 通过GRU层
output, h_n = gru(X)

# 输出形状
print("output shape:", output.shape)  # (5, 3, 20)
print("h_n type:", h_n[0].shape)  # (2, 3, 20)