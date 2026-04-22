"""
Module for attention mechanysm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention.
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Q: [batch, num_heads, seq_len_q, d_k]
    K: [batch, num_heads, seq_len_k, d_k]
    V: [batch, num_heads, seq_len_k, d_v]
    mask: [batch, 1, seq_len_q, seq_len_k]
    """
    
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, mask=None):
        # Apply attention formula
        scores = Q @ K.transpose(-2, -1)                    # [batch, num_heads, seq_len_q, seq_len_k]
        scores = scores / math.sqrt(Q.shape[-1])            # Q.shape[-1] == d_k in the paper

        # Check if there's a mask and apply masking if any
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)    # -1e9 to make the values 0 after softmax where approximate to -inf
                                                            # Not float('-inf') since it can create NaN values

        # Apply softmax and dropout for attention
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Calculate the output
        output = attn @ V                                   # [batch, num_heads, seq_len_q, d_v]

        return output, attn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention.
    
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Size of every head

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)
    
    def forward(self, Q, K, V, mask=None):
        # Linear projection
        Q = self.W_Q(Q)  # [batch, seq_len, d_model]
        K = self.W_K(K)
        V = self.W_V(V)

        # Divide to heads (reshape + transpose)
        batch_size = Q.size(0)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)                        # Q shape: [batch, num_heads, seq_len_q, d_k]
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply attention
        attn_output, attn_weights = self.attention(Q, K, V, mask)                                   # attn_output shape: [batch, num_heads, seq_len_q, d_k]

        # Unify the heads (transpose + reshape)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)   # [batch, seq_len_q, d_model]

        # Last projection
        output = self.W_O(attn_output)

        return output, attn_weights


if __name__ == '__main__':
    batch, seq_len, d_model, num_heads = 2, 10, 512, 8
    
    # Test 1: Shape correctness
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    mha.eval()
    x = torch.randn(batch, seq_len, d_model)
    out, attn = mha(x, x, x)
    assert out.shape == (batch, seq_len, d_model), f"Expected {(batch, seq_len, d_model)}, got {out.shape}"
    assert attn.shape == (batch, num_heads, seq_len, seq_len), f"Attn shape wrong: {attn.shape}"
    print("Test 1 passed: shapes correct")
    
    # Test 2: Attention weights sum to 1 (probability distribution)
    attn_sum = attn.sum(dim=-1)
    assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5), "Attention weights don't sum to 1"
    print("Test 2 passed: attention weights are valid probability distribution")
    
    # Test 3: Masking works
    mask = torch.ones(batch, 1, 1, seq_len)
    mask[:, :, :, 5:] = 0  # mask last 5 positions
    out_masked, attn_masked = mha(x, x, x, mask=mask)
    
    # Attention weights of masked positions should be ~0
    assert attn_masked[:, :, :, 5:].abs().max() < 1e-5, "Masked positions have non-zero attention"
    print("Test 3 passed: masking works correctly")
    
    # Test 4: Cross-attention with different seq lengths
    q = torch.randn(batch, 7, d_model)  # decoder query
    kv = torch.randn(batch, 12, d_model)  # encoder key/value
    out_cross, attn_cross = mha(q, kv, kv)
    assert out_cross.shape == (batch, 7, d_model), f"Cross-attention shape wrong: {out_cross.shape}"
    assert attn_cross.shape == (batch, num_heads, 7, 12), f"Cross-attention attn shape wrong: {attn_cross.shape}"
    print("Test 4 passed: cross-attention works with different Q/K lengths")
    
    print("\nAll tests passed!")