import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.embed_dim = embed_dim  # 即d_model
        self.num_heads = num_heads  # 即注意力头数
        self.head_dim = embed_dim // num_heads  # 每个头的维度
        self.dropout = dropout
        assert self.head_dim * num_heads == embed_dim

        # 初始化W_Q,W_K,W_V,W_O
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """
        Args:
            query: (n, N, embed_dim)
            key: (m, N, embed_dim)
            value: (m, N, embed_dim)
            attn_mask (bool Tensor or float Tensor): (n, m) or (N * num_heads, n, m)
            key_padding_mask (bool Tensor): (N, m)

        Returns:
            attn_output: (n, N, embed_dim)
            attn_output_weights: (N, num_heads, n, m)
        """
        return self._multi_head_forward_attention(query,
                                                  key,
                                                  value,
                                                  dropout_p=self.dropout,
                                                  attn_mask=attn_mask,
                                                  key_padding_mask=key_padding_mask,
                                                  training=self.training)

    def _multi_head_forward_attention(self,
                                      query,
                                      key,
                                      value,
                                      dropout_p,
                                      attn_mask=None,
                                      key_padding_mask=None,
                                      training=True):
        ############################
        # 第一阶段: 计算投影后的Q, K, V
        ############################
        q = self.q_proj(query)  # (n, N, embed_dim)
        k = self.k_proj(key)  # (m, N, embed_dim)
        v = self.v_proj(value)  # (m, N, embed_dim)

        ############################
        # 第二阶段: attn_mask的维度检查
        ############################
        n, N, embed_dim = q.size()
        m = key.size(0)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                if attn_mask.shape != (n, m):
                    raise RuntimeError
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                if attn_mask.shape != (self.num_heads * N, n, m):
                    raise RuntimeError
            else:
                raise RuntimeError

        ##########################################
        # 第三阶段: 将attn_mask和key_padding_mask合并
        ##########################################
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (N, m)
            key_padding_mask = key_padding_mask.view(N, 1, 1, m).expand(-1, self.num_heads, -1,
                                                                        -1).reshape(self.num_heads * N, 1, m)
            if attn_mask is None:
                attn_mask = key_padding_mask
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.logical_or(key_padding_mask)
            else:
                attn_mask = attn_mask.masked_fill(key_padding_mask, float('-inf'))

        # 将attn_mask转换成浮点型张量
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            new_attn_mask.masked_fill_(attn_mask, float('-inf'))
            attn_mask = new_attn_mask

        ###################
        # 第四阶段: 计算注意力
        ###################
        # 将多头注意力化简为高维单头注意力
        q = q.reshape(n, N * self.num_heads, self.head_dim).transpose(0, 1)  # (N * num_heads, n, head_dim)
        k = k.reshape(m, N * self.num_heads, self.head_dim).transpose(0, 1)  # (N * num_heads, m, head_dim)
        v = v.reshape(m, N * self.num_heads, self.head_dim).transpose(0, 1)  # (N * num_heads, m, head_dim)

        if not training:
            dropout_p = 0.0

        attn_output, attn_output_weights = self._scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
        # 截至目前，attn_output: (N * num_heads, n, head_dim), attn_output_weights: (N * num_heads, n, m)
        attn_output = attn_output.transpose(0, 1).reshape(n, N, embed_dim)  # 合并num_heads个头的结果
        attn_output = self.out_proj(attn_output)
        attn_output_weights = attn_output_weights.reshape(N, self.num_heads, n, m)
        return attn_output, attn_output_weights

    def _scaled_dot_product_attention(self, q, k, v, attn_mask=None, dropout_p=0.0):
        """
        Args:
            q: (N, n, E), where E is embedding dimension.
            k: (N, m, E)
            v: (N, m, E)
            attn_mask: (n, m) or (N, n, m)
        
        Returns:
            attn_output: (N, n, E)
            attn_weights: (N, n, m)
        """
        q = q / math.sqrt(q.size(2))
        if attn_mask is not None:
            scores = q @ k.transpose(-2, -1) + attn_mask
        else:
            scores = q @ k.transpose(-2, -1)

        attn_weights = F.softmax(scores, dim=-1)
        if dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=dropout_p)
        attn_output = attn_weights @ v
        return attn_output, attn_weights
