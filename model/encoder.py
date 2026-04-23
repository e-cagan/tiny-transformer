"""
Module for encoder.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadAttention


class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # Calculate positional encoding
        # Create PE matrix
        pe = torch.zeros(max_len, d_model)

        # Find position vector and freq term
        position = torch.arange(0, max_len).unsqueeze(1).float()                                    # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))  # [d_model/2]

        # Apply sin (for even indicies) and cos (for odd indicies) in PE matrix
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Unsqueeze for batch dimension
        pe = pe.unsqueeze(0)                                                                        # [1, max_len, d_model]

        # Save PE matrix inside of the buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch, seq_len, d_model]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class PositionwiseFeedForward(nn.Module):
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()

        # Layers
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Feed forward neural network
        x = self.linear_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)

        return x


class EncoderLayer(nn.Module):
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # Layers
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Neural network
        attn_out, _ = self.self_attn(x, x, x, mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class Encoder(nn.Module):
    
    def __init__(self, src_vocab_size, d_model, num_layers, num_heads, d_ff, max_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Layers
        self.emb = nn.Embedding(src_vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_len)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, mask=None):
        # src: [batch, seq_len] — token IDs (long tensor)
        # mask: [batch, 1, 1, seq_len] or None
        
        # Embedding + scale
        x = self.emb(src) * math.sqrt(self.d_model)
        
        # Positional encoding
        x = self.pe(x)
        
        # Dropout
        x = self.dropout(x)
        
        # Propagate the input from layers in order
        for layer in self.encoder_layers:
            x = layer(x, mask)
        
        return x


if __name__ == '__main__':
    batch, seq_len = 2, 10
    src_vocab_size, d_model = 1000, 512
    num_layers, num_heads, d_ff = 6, 8, 2048

    # Test 1: PFF
    pff = PositionwiseFeedForward(d_model, d_ff)
    x = torch.randn(batch, seq_len, d_model)
    assert pff(x).shape == x.shape
    print("Test 1: PFF ok")

    # Test 2: PE
    pe = PositionalEncoding(d_model)
    out = pe(x)
    assert out.shape == x.shape
    assert not torch.equal(out, x), "PE didn't change anything"
    print("Test 2: PE ok")

    # Test 3: EncoderLayer
    layer = EncoderLayer(d_model, num_heads, d_ff)
    assert layer(x).shape == x.shape
    print("Test 3: EncoderLayer ok")

    # Test 4: Encoder
    encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, d_ff, max_len=5000)
    src = torch.randint(0, src_vocab_size, (batch, seq_len))
    out = encoder(src)
    assert out.shape == (batch, seq_len, d_model)
    print("Test 4: Encoder ok")

    # Test 5: Mask
    mask = torch.ones(batch, 1, 1, seq_len)
    mask[:, :, :, 7:] = 0
    out_masked = encoder(src, mask)
    assert out_masked.shape == (batch, seq_len, d_model)
    print("Test 5: Mask ok")

    # Test 6: Param count
    param_count = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder parameter count: {param_count:,}")  # expected ~19-20M