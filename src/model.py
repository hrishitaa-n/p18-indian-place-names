
import math, torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class Config:
    vocab_size: int = 30
    block_size: int = 32
    n_layer: int = 4
    n_head: int = 6
    n_embd: int = 192
    dropout: float = 0.1

def sinusoidal(T, D):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017)."""
    pos = torch.arange(T).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, D, 2).float() * (-math.log(10000.0) / D))
    pe = torch.zeros(T, D)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe

class CausalSelfAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.h = c.n_head
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd)
        self.proj = nn.Linear(c.n_embd, c.n_embd)
        self.drop = nn.Dropout(c.dropout)
        # lower-triangular mask: position t may only look at positions <= t
        m = torch.tril(torch.ones(c.block_size, c.block_size)).bool()
        self.register_buffer("mask", m.view(1, 1, c.block_size, c.block_size))

    def forward(self, x, return_att=False):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = [t.view(B, T, self.h, C // self.h).transpose(1, 2) for t in (q, k, v)]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(C // self.h)
        att = att.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (self.drop(att) @ v).transpose(1, 2).reshape(B, T, C)
        y = self.drop(self.proj(y))
        return (y, att) if return_att else y

class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(c.n_embd), nn.LayerNorm(c.n_embd)
        self.attn = CausalSelfAttention(c)
        self.mlp = nn.Sequential(nn.Linear(c.n_embd, 4 * c.n_embd), nn.GELU(),
                                 nn.Linear(4 * c.n_embd, c.n_embd), nn.Dropout(c.dropout))
    def forward(self, x, return_att=False):
        if return_att:
            a, att = self.attn(self.ln1(x), True)
            x = x + a
            return x + self.mlp(self.ln2(x)), att
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))

class MiniGPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab_size, c.n_embd)
        self.register_buffer("pe", sinusoidal(c.block_size, c.n_embd))
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layer)])
        self.ln_f = nn.LayerNorm(c.n_embd)
        self.head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.head.weight = self.tok.weight          # weight tying
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None, return_att=False):
        B, T = idx.shape
        x = self.drop(self.tok(idx) * math.sqrt(self.c.n_embd) + self.pe[:T])
        atts = []
        for b in self.blocks:
            if return_att:
                x, a = b(x, True); atts.append(a)
            else:
                x = b(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        if return_att:
            return logits, loss, atts
        return logits, loss

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
