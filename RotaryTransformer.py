import torch
from typing import Optional, Tuple
from torch import nn
import math
import torch.nn.functional as F

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x).type_as(x)
        return output * self.weight


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def precompute_freqs_cis(dim, end, device, theta = 10000.0,):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)] / dim))
    t = torch.arange(end)  # type: ignore
    freqs = torch.outer(t, freqs)  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    freqs_cis = freqs_cis.to(device)
    return freqs_cis


class AttentionWithRotary(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = self.dim // self.n_heads

        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=True)
        self.wk = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=True)
        self.wv = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=True)
        self.wo = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=True)

    def apply_rotary_emb(self, xq, xk, freqs_cis):
        xq_ = torch.view_as_complex(xq.reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.reshape(*xk.shape[:-1], -1, 2))
        freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)

    def forward(self, x, freqs_cis):
        batch_size = x.size(0)
        seq_len = x.size(1)
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_heads, self.head_dim)
        xq, xk = self.apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        query = xq.transpose(1, 2)
        keys = xk.transpose(1, 2)
        values = xv.transpose(1, 2)
        scores = torch.matmul(query, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.wo(output)
        return output

class SwiFeedForward(nn.Module):
    def __init__(self,dim, hidden_dim, multiple_of):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        output = self.w2(F.silu(self.w1(x)) * self.w3(x))
        return output

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads ,norm_eps):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = self.dim // self.n_heads
        self.norm_eps = norm_eps
        self.attention = AttentionWithRotary(self.dim, self.n_heads)
        self.feed_forward = SwiFeedForward(dim, dim * 4, int(dim/2))
        self.attention_norm = RMSNorm(self.dim, eps=self.norm_eps)
        self.ffn_norm = RMSNorm (self.dim, eps=self.norm_eps)

    def forward(self, x, freqs_cis):
        h = x + self.attention.forward(self.attention_norm(x), freqs_cis)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out

class TransformerRotatry(nn.Module):
    def __init__(self, dim, n_heads, seq_len ,norm_eps, n_layers = 2):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.seq_len = seq_len
        self.head_dim = self.dim // self.n_heads
        self.norm_eps = norm_eps
        self.n_layers = n_layers
        self.layers = torch.nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(TransformerBlock(dim, n_heads ,norm_eps))
        self.norm = RMSNorm(self.dim, eps=self.norm_eps)
        self.output = nn.Linear(self.dim, self.dim, bias=False)
        self.freqs_cis = None

    def forward(self, x):
        if self.freqs_cis == None:
            self.freqs_cis = precompute_freqs_cis(dim = self.head_dim, end = self.seq_len, device = x.device)
        h = x
        for layer in self.layers:
            h = layer(h, self.freqs_cis)
        h = self.norm(h)
        output = self.output(h[:, -1, :])  # only compute last logits
        return output