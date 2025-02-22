import torch
from d2l import torch as d2l

X, W_xh = torch.normal(0, 1, (3, 1)), torch.normal(0, 1, (1, 4))
H, W_hh = torch.normal(0, 1, (3, 4)), torch.normal(0, 1, (4, 4))
A = torch.matmul(X, W_xh) + torch.matmul(H, W_hh)
B = torch.matmul(torch.cat((X, H), 1), torch.cat((W_xh, W_hh), 0))
print(A)
print(B) 

