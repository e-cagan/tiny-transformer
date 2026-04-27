"""
Visualize cross-attention heatmap for a single translation.
Usage: python visualize.py "A sentence to visualize."
"""

import sys
import os
import math
import torch
import matplotlib.pyplot as plt
import numpy as np

from dataset import Vocabulary, tokenize_en
from model.transformer import Transformer, make_src_mask, make_tgt_mask
from train import config
from translate import build_vocabs, load_model


# ============================================================
# Hook setup
# ============================================================

# Module-level dict to store attention weights captured by hooks
attention_storage = {}


def make_hook(name):
    """
    Factory that creates a forward hook capturing the attention
    weights from a cross_attn module's output.
    
    The closure 'name' lets each hook know which layer it belongs to.
    """
    
    def hook(module, input, output):
        # MultiHeadAttention.forward returns (attn_output, attn_weights)
        # We want attn_weights, which is output[1]
        # .detach() removes from grad graph, .cpu() moves to CPU for plotting
        attention_storage[name] = output[1].detach().cpu()
    
    return hook


def register_hooks(model):
    """
    Attach forward hooks to every decoder layer's cross_attn module.
    Returns: list of hook handles for later removal.
    """
    
    handles = []
    for i, layer in enumerate(model.decoder.decoder_layers):
        handle = layer.cross_attn.register_forward_hook(make_hook(f"layer_{i}"))
        handles.append(handle)
    
    return handles


def remove_hooks(handles):
    """Remove all registered hooks. Always do this when done."""
    
    for handle in handles:
        handle.remove()


# ============================================================
# Translation with attention capture
# ============================================================

def translate_and_capture(model, src_vocab, tgt_vocab, sentence, device, max_len=50):
    """
    Translate a sentence and capture the cross-attention weights.
    
    Returns:
      pred_tokens: list of predicted German tokens (no <sos>/<eos>)
      src_tokens_with_eos: list of source tokens including <eos> (for plotting)
      attention: tensor of shape [num_heads, tgt_len, src_len] from last decoder layer
    """
    
    model.eval()
    
    # Source preparation 
    tokens = tokenize_en(sentence)
    src_tokens_with_eos = tokens + [Vocabulary.EOS_TOKEN]  # for plot x-axis
    src_ids = src_vocab.encode(src_tokens_with_eos)
    src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, src_len]
    
    # Target initialization 
    tgt_ids = [tgt_vocab.sos_idx]
    
    with torch.no_grad():
        # Encoder runs once
        src_mask = make_src_mask(src, pad_idx=0)
        enc_out = model.encoder(src, src_mask)
        
        # Autoregressive loop — each iteration triggers the hooks
        for _ in range(max_len):
            tgt = torch.tensor(tgt_ids, dtype=torch.long, device=device).unsqueeze(0)
            tgt_mask = make_tgt_mask(tgt, pad_idx=0)
            
            # This call triggers all cross_attn hooks
            # attention_storage["layer_N"] gets overwritten each iteration
            dec_out = model.decoder(tgt, enc_out, src_mask, tgt_mask)
            
            next_token = dec_out[0, -1].argmax().item()
            tgt_ids.append(next_token)
            
            if next_token == tgt_vocab.eos_idx:
                break
    
    # After the loop, attention_storage holds the FINAL iteration's attention,
    # which is the full [1, num_heads, tgt_len, src_len] matrix for the
    # complete generated sequence.
    num_layers = len(model.decoder.decoder_layers)
    last_layer_attn = attention_storage[f"layer_{num_layers - 1}"]  # [1, h, tgt, src]
    last_layer_attn = last_layer_attn.squeeze(0)                    # [h, tgt, src]
    
    # Decode predicted tokens (skip <sos> and possibly <eos>)
    result_ids = tgt_ids[1:]
    if result_ids and result_ids[-1] == tgt_vocab.eos_idx:
        result_ids = result_ids[:-1]
    pred_tokens = tgt_vocab.decode(result_ids)
    
    return pred_tokens, src_tokens_with_eos, last_layer_attn


# ============================================================
# Plotting
# ============================================================

def plot_attention_heatmap(attention, src_tokens, tgt_tokens, save_path, head=None):
    """
    Plot the cross-attention as a heatmap.
    
    attention: [num_heads, tgt_len, src_len]
    src_tokens: list of source tokens (x-axis)
    tgt_tokens: list of predicted target tokens (y-axis)
    head: int → plot that head only; None → average across heads
    """
    
    if head is None:
        attn_matrix = attention.mean(dim=0).numpy()  # [tgt_len, src_len]
        title_suffix = "head-averaged"
    else:
        attn_matrix = attention[head].numpy()
        title_suffix = f"head {head}"
    
    # Trim attention matrix to actual token lengths
    # (in case attention has extra rows/cols from <sos> handling)
    tgt_len = len(tgt_tokens)
    src_len = len(src_tokens)
    attn_matrix = attn_matrix[:tgt_len, :src_len]
    
    # Figure size scales with sequence length
    fig_w = max(8, src_len * 0.6)
    fig_h = max(6, tgt_len * 0.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    
    im = ax.imshow(attn_matrix, cmap='viridis', aspect='auto')
    
    ax.set_xticks(range(src_len))
    ax.set_xticklabels(src_tokens, rotation=45, ha='right')
    ax.set_yticks(range(tgt_len))
    ax.set_yticklabels(tgt_tokens)
    
    ax.set_xlabel('Source (English)')
    ax.set_ylabel('Target (German)')
    ax.set_title(f'Cross-Attention Heatmap (last layer, {title_suffix})')
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    if len(sys.argv) < 2:
        sentence = "A man is walking in the park."
    else:
        sentence = ' '.join(sys.argv[1:])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load vocabs and model (reuse helpers from translate.py)
    src_vocab, tgt_vocab = build_vocabs()
    model = load_model('checkpoints/best_model_epoch18.pt', len(src_vocab), len(tgt_vocab), device)
    
    # Register hooks BEFORE translation
    handles = register_hooks(model)
    
    try:
        # Translate; hooks fire automatically on every decoder forward
        pred_tokens, src_tokens, attention = translate_and_capture(
            model, src_vocab, tgt_vocab, sentence, device
        )
    finally:
        # ALWAYS clean up hooks, even if something went wrong
        remove_hooks(handles)
    
    print(f"Input:  {sentence}")
    print(f"Output: {' '.join(pred_tokens)}")
    print(f"Attention shape: {attention.shape}  [num_heads, tgt_len, src_len]")
    
    # Save heatmap
    os.makedirs('figures', exist_ok=True)
    plot_attention_heatmap(
        attention, src_tokens, pred_tokens,
        save_path='figures/attention_heatmap.png',
        head=None  # average across all heads
    )
    print(f"Saved heatmap to figures/attention_heatmap.png")


if __name__ == '__main__':
    main()