"""
Inference script for trained Transformer.
Usage: python translate.py "A sentence in English."
"""

import sys
import torch
from datasets import load_dataset

from dataset import Vocabulary, Multi30kDataset, tokenize_en, tokenize_de
from model.transformer import Transformer, make_src_mask, make_tgt_mask
from train import config, translate_sample


def load_model(checkpoint_path, src_vocab_size, tgt_vocab_size, device):
    """Load trained model from checkpoint."""
    
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        max_len=config['max_len'],
        dropout=config['dropout'],
        pad_idx=config['pad_idx'],
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']+1}, val_loss={checkpoint['val_loss']:.4f}")
    return model


def build_vocabs():
    """Rebuild vocabularies from train set (slow but simple)."""
    
    print("Building vocabularies...")
    ds = load_dataset("bentrevett/multi30k")
    train = ds['train']
    
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    src_vocab.build([ex['en'] for ex in train], tokenize_en, min_freq=config['min_freq'])
    tgt_vocab.build([ex['de'] for ex in train], tokenize_de, min_freq=config['min_freq'])
    
    return src_vocab, tgt_vocab


def main():
    if len(sys.argv) < 2:
        print('Usage: python translate.py "Sentence to translate"')
        sys.exit(1)
    
    sentence = ' '.join(sys.argv[1:])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Build vocabs
    src_vocab, tgt_vocab = build_vocabs()
    
    # Load model
    model = load_model(
        'checkpoints/best_model_epoch18.pt',
        len(src_vocab),
        len(tgt_vocab),
        device
    )
    
    # Translate
    translation = translate_sample(model, src_vocab, tgt_vocab, sentence, device)
    print(f"\nInput:  {sentence}")
    print(f"Output: {translation}")


if __name__ == '__main__':
    main()