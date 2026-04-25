"""
Module for training the Transformer model.
"""

import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset

from dataset import Vocabulary, Collate, Multi30kDataset, tokenize_en, tokenize_de
from model.transformer import Transformer


# ============================================================
# Config
# ============================================================
# hyperparameters (batch_size, lr, epochs, d_model, num_layers, etc.)
config = {
    'd_model': 512,
    'num_layers': 6,
    'num_heads': 8,
    'd_ff': 2048,
    'dropout': 0.1,
    'max_len': 5000,
    
    'batch_size': 32,
    'num_epochs': 20,
    'warmup_steps': 4000,
    'label_smoothing': 0.1,
    'clip_grad': 1.0,
    
    'adam_betas': (0.9, 0.98),
    'adam_eps': 1e-9,
    
    'pad_idx': 0,
    'min_freq': 2,
    'max_seq_len': 50,
}

# ============================================================
# Data preparation
# ============================================================

def prepare_data(config):
    """
    Load Multi30k, build vocabularies, create train/val datasets and loaders.
    Returns: (train_loader, val_loader, src_vocab, tgt_vocab)
    """
    
    # load dataset
    ds = load_dataset("bentrevett/multi30k")
    train = ds['train']
    val = ds['validation']

    # build src and tgt vocabularies from train set
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    src_vocab.build([ex['en'] for ex in train], tokenize_en, min_freq=config['min_freq'])
    tgt_vocab.build([ex['de'] for ex in train], tokenize_de, min_freq=config['min_freq'])

    # create Dataset instances for train and val
    train_ds = Multi30kDataset(train, src_tokenizer=tokenize_en, tgt_tokenizer=tokenize_de, src_vocab=src_vocab, tgt_vocab=tgt_vocab, max_len=config['max_seq_len'])
    val_ds = Multi30kDataset(val, src_tokenizer=tokenize_en, tgt_tokenizer=tokenize_de, src_vocab=src_vocab, tgt_vocab=tgt_vocab, max_len=config['max_seq_len'])

    # create Collate function with pad_idx
    collate_fn = Collate(config['pad_idx'])

    # create DataLoaders
    train_dl = DataLoader(train_ds, config['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, config['batch_size'], shuffle=False, collate_fn=collate_fn)

    return train_dl, val_dl, src_vocab, tgt_vocab


# ============================================================
# Learning rate schedule
# ============================================================

def noam_schedule(step, d_model, warmup_steps):
    """
    Paper's learning rate schedule:
    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    """
    
    # Implement the formula
    lr = d_model**(-0.5) * min(step**(-0.5), step * warmup_steps**(-1.5))
    return lr


# ============================================================
# Training
# ============================================================

def train_epoch(model, loader, criterion, optimizer, scheduler, device, clip_grad=1.0):
    """
    One training epoch.
    Returns: average train loss
    """
    
    # Set model to train mode
    model.train()
    total_loss = 0.0
    
    # tqdm visualizer
    pbar = tqdm(loader, desc="Training")
    
    # Iterate trough batches (src, tgt)
    for src, tgt in pbar:
        src = src.to(device)
        tgt = tgt.to(device)
        
        # Teacher forcing shift
        decoder_input = tgt[:, :-1]
        target = tgt[:, 1:]
        
        # Make gradients zero
        optimizer.zero_grad()
        
        # Forward propagation
        outputs = model(src, decoder_input)
        
        # Calculate loss
        loss = criterion(
            outputs.reshape(-1, outputs.size(-1)),
            target.reshape(-1)
        )
        
        # Backward propagation
        loss.backward()
        
        # Gradient clip + step
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    return total_loss / len(loader)


# ============================================================
# Validation
# ============================================================

def validate(model, loader, criterion, device):
    """
    Run validation. No gradient computation.
    Returns: average val loss
    """
    # TODO: model.eval()
    # TODO: torch.no_grad() context
    # TODO: loop over batches, compute loss, accumulate
    pass


# ============================================================
# Sample translation (sanity check)
# ============================================================

def translate_sample(model, src_vocab, tgt_vocab, sentence, device, max_len=50):
    """
    Greedy decode a single sentence. For qualitative monitoring.
    """
    # TODO: tokenize, encode, add <sos>/<eos>
    # TODO: encoder forward
    # TODO: autoregressive decoding loop
    # TODO: decode IDs back to tokens
    pass


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(model, optimizer, epoch, val_loss, path):
    """Save model, optimizer state, metadata."""
    # TODO
    pass


# ============================================================
# Main
# ============================================================

def main():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = {
        # TODO: fill
    }
    
    # Prepare data
    # TODO
    
    # Create model
    # TODO
    
    # Loss, optimizer, scheduler
    # TODO
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        # TODO: train_epoch
        # TODO: validate
        # TODO: log losses
        # TODO: translate_sample (every N epochs)
        # TODO: checkpoint if val_loss improved
        pass
    
    # Save loss curve
    # TODO: matplotlib plot train_losses and val_losses
    

if __name__ == '__main__':
    main()