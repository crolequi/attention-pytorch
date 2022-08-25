import torch.nn as nn
from attn import ScaledDotProductAttention


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, key_size, value_size, dropout=0):
        super().__init__()
        self.attn = ScaledDotProductAttention(dropout)
        self.W_q = nn.Linear(embed_dim, key_size, bias=False)
        self.W_k = nn.Linear(embed_dim, key_size, bias=False)
        self.W_v = nn.Linear(embed_dim, value_size, bias=False)

    def forward(self, X, attn_mask=None):
        """
        Args:
            X: input sequence, shape: (N, L, embed_dim)
            attn_mask: (N, L, L)
        """
        query = self.W_q(X)  # (N, L, key_size)
        key = self.W_k(X)  # (N, L, key_size)
        value = self.W_v(X)  # (N, L, value_size)
        return self.attn(query, key, value, attn_mask)  # (N, L, value_size)
