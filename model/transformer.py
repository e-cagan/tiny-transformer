"""
Module for transformer architecture.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import Encoder
from .decoder import generate_casual_mask, Decoder


# Helper functions
def make_src_mask(src, pad_idx):
    """
    Masks padding positions in encoder.

    src shape: [batch, src_len]

    pad_idx -> scaler
    """

    # Apply masking
    pad_mask = src != pad_idx

    # Unsqueeze to obtaing expected shape which is [batch, 1, 1, src_len]
    pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)

    return pad_mask


def make_tgt_mask(tgt, pad_idx):
    """
    Masks padding positions and futures in decoder.

    tgt shape: [batch, tgt_len]

    pad_idx -> scaler
    """

    # Create padding mask
    pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)               # [batch, 1, 1, tgt_len]

    # Generate casual mask for futures
    casual_mask = generate_casual_mask(tgt.size(1), device=tgt.device)  # [1, 1, tgt_len, tgt_len]

    # Unify the masks to obtain target mask
    tgt_mask = pad_mask & casual_mask                                   # [batch, 1, tgt_len, tgt_len]

    return tgt_mask


class Transformer(nn.Module):

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_layers=6, num_heads=8, d_ff=2048, max_len=5000, dropout=0.1, pad_idx=0):
        super().__init__()

        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout)
        self.pad_idx = pad_idx

    def forward(self, src, tgt):
        # src: [batch, src_len]
        # tgt: [batch, tgt_len]
        
        src_mask = make_src_mask(src, self.pad_idx)
        tgt_mask = make_tgt_mask(tgt, self.pad_idx)
        
        enc_out = self.encoder(src, src_mask)
        logits = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        
        return logits


if __name__ == '__main__':
    batch, src_len, tgt_len = 2, 10, 8
    src_vocab_size, tgt_vocab_size = 1000, 1200
    
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=512, num_layers=6, num_heads=8, d_ff=2048,
        pad_idx=0
    )
    
    src = torch.randint(1, src_vocab_size, (batch, src_len))  # 1'den başla, 0=pad
    tgt = torch.randint(1, tgt_vocab_size, (batch, tgt_len))
    
    # Bir iki padding ekle (mask testi için)
    src[0, -2:] = 0
    tgt[1, -1] = 0
    
    # Test 1: Forward
    logits = model(src, tgt)
    assert logits.shape == (batch, tgt_len, tgt_vocab_size), f"got {logits.shape}"
    print("Test 1: forward shape ok")
    
    # Test 2: NaN check
    assert not torch.isnan(logits).any(), "NaN in logits"
    print("Test 2: no NaN")
    
    # Test 3: Backward pass
    loss = logits.sum()
    loss.backward()
    # Gradyanların hepsi hesaplanmış mı?
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} has no grad"
    print("Test 3: backward pass ok")
    
    # Test 4: Param count
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Total parameter count: {param_count:,}")  # ~45-50M bekle
    
    # Test 5: Mask fonksiyonları
    src_mask = make_src_mask(src, 0)
    assert src_mask.shape == (batch, 1, 1, src_len)
    assert src_mask[0, 0, 0, -1] == False  # son iki padding'di
    assert src_mask[0, 0, 0, 0] == True    # ilk token padding değil
    print("Test 5: src mask ok")
    
    tgt_mask = make_tgt_mask(tgt, 0)
    assert tgt_mask.shape == (batch, 1, tgt_len, tgt_len)
    # Causal: pozisyon 0 pozisyon 1'i göremesin
    assert tgt_mask[0, 0, 0, 1] == False
    # Causal: pozisyon 1 pozisyon 0'ı görebilsin (padding yoksa)
    assert tgt_mask[0, 0, 1, 0] == True
    print("Test 6: tgt mask ok")
    
    print("\nAll tests passed!")