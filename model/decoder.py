"""
Module for decoder.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadAttention
from encoder import PositionalEncoding, PositionwiseFeedForward


class DecoderLayer(nn.Module):
    """Masked self-attention + Cross-attention + FFN"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # Layers
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        # Neural network

        # Ignore the weights in attentions
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask=tgt_mask)[0]))
        x = self.norm2(x + self.dropout(self.cross_attn(x, enc_out, enc_out, mask=src_mask)[0]))
        x = self.norm3(x + self.dropout(self.ffn(x)))

        return x

class Decoder(nn.Module):
    """Embedding + PE + N * DecoderLayer + output projection"""

    def __init__(self, tgt_vocab_size, d_model, num_layers, num_heads, d_ff, max_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Layers
        self.emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_len)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, tgt, enc_out, src_mask=None, tgt_mask=None):
        # Neural network
        
        # Embedding + scale
        x = self.emb(tgt) * math.sqrt(self.d_model)

        # Positional encoding
        x = self.pe(x)

        # Dropout
        x = self.dropout(x)

        # Propagate the input from layers in order
        for layer in self.decoder_layers:
            x = layer(x, enc_out, src_mask, tgt_mask)

        # Output projection
        x = self.output_projection(x)

        return x


def generate_causal_mask(seq_len):
    """Returns [1, 1, seq_len, seq_len] lower-triangular mask (1 = visible, 0 = masked)"""
    
    # Take the lower triangle of matrix
    mask = torch.tril(torch.ones(seq_len, seq_len)) # shape: [seq_len, seq_len]
    
    # Unsqueeze twice to obtain the [1, 1, seq_len, seq_len] shape
    mask = mask.unsqueeze(0).unsqueeze(0)
    
    return mask


if __name__ == '__main__':
    batch, src_len, tgt_len = 2, 10, 8
    src_vocab_size, tgt_vocab_size, d_model = 1000, 1200, 512
    num_layers, num_heads, d_ff = 6, 8, 2048
    
    # Dummy encoder output
    enc_out = torch.randn(batch, src_len, d_model)
    tgt = torch.randint(0, tgt_vocab_size, (batch, tgt_len))
    
    # Test 1: Causal mask
    causal = generate_causal_mask(tgt_len)
    assert causal.shape == (1, 1, tgt_len, tgt_len), f"got {causal.shape}"
    assert causal[0, 0, 0, 1] == 0, "position 0 should not see position 1"
    assert causal[0, 0, 1, 0] == 1, "position 1 should see position 0"
    assert causal[0, 0, 3, 3] == 1, "diagonal should be 1"
    print("Test 1: causal mask ok")
    
    # Test 2: DecoderLayer
    layer = DecoderLayer(d_model, num_heads, d_ff)
    x = torch.randn(batch, tgt_len, d_model)
    out = layer(x, enc_out, src_mask=None, tgt_mask=causal)
    assert out.shape == (batch, tgt_len, d_model)
    print("Test 2: DecoderLayer ok")
    
    # Test 3: Decoder (full)
    decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, d_ff)
    logits = decoder(tgt, enc_out, src_mask=None, tgt_mask=causal)
    assert logits.shape == (batch, tgt_len, tgt_vocab_size), f"got {logits.shape}"
    print("Test 3: Decoder ok")
    
    # Test 4: Masking works (causal test)
    tgt1 = tgt.clone()
    tgt2 = tgt.clone()
    tgt2[:, -1] = (tgt2[:, -1] + 1) % tgt_vocab_size  # Change the last token
    
    decoder.eval()  # close the dropout to protect the amount
    logits1 = decoder(tgt1, enc_out, src_mask=None, tgt_mask=causal)
    logits2 = decoder(tgt2, enc_out, src_mask=None, tgt_mask=causal)
    
    # Leak info test
    assert torch.allclose(logits1[:, :-1], logits2[:, :-1], atol=1e-5), "causal mask leaking future info"
    print("Test 4: causal masking works correctly")
    
    # Test 5: Param count
    param_count = sum(p.numel() for p in decoder.parameters())
    print(f"Decoder parameter count: {param_count:,}")  # ~26-28M