import torch
import torch.nn as nn

from torch.nn import functional as F


torch.manual_seed(1337)

class Head(nn.Module):
    def __init__(self, embd_size, head_size):
        super().__init__()
        self.key = nn.Linear(embd_size, head_size, bias=False)
        self.query = nn.Linear(embd_size, head_size, bias=False)
        self.value = nn.Linear(embd_size, head_size, bias=False)

    def forward(self, k, q, v, mask=None):
        k = self.key(k)
        q = self.query(q)
        v = self.value(v)

        B,T,C = k.shape # C is head_size

        weights = q @ k.transpose(-2,-1) * C**-0.5

        if mask is not None:
            self.register_buffer('tril', torch.tril(torch.ones(T, T)))
            weights = weights.masked_fill(self.tril == 0, float('-inf'))

        weights = F.softmax(weights, dim=-1)

        out = weights @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, embd_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(embd_size, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, embd_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, k, q, v, mask=None):
        out = torch.cat([head(k, q, v, mask=mask) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, embd_size, dropout):
        super().__init__()
        self.fc_block = nn.Sequential(
            nn.Linear(embd_size, 4 * embd_size),
            nn.ReLU(),
            nn.Linear(4 * embd_size, embd_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.fc_block(x)


        
