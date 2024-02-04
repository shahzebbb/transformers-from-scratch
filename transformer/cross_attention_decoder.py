from .utils import MultiHeadAttention, FeedForward
import torch.nn as nn

class CrossAttentionDecoderBlock(nn.Module):
    def __init__(self, embd_size, num_heads, dropout):
        super().__init__()
        head_size = embd_size // num_heads
        self.mha = MultiHeadAttention(num_heads, head_size, embd_size, dropout)
        self.ff = FeedForward(embd_size, dropout)
        self.ln1 = nn.LayerNorm(embd_size)
        self.ln2 = nn.LayerNorm(embd_size)

    def forward(self, enc, dec, mask = True):
        _x = dec
        enc = self.ln1(enc)
        dec = self.ln1(dec)
        x = _x + self.mha(k=enc, q=dec, v=enc, mask=mask)
        x = x + self.ff(self.ln2(x))
        return x