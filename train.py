"""
Module for training the Transformer model.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset

from dataset import Vocabulary, Collate, Multi30kDataset, tokenize_en, tokenize_de
from model.transformer import Transformer, make_src_mask, make_tgt_mask


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

    # build src and tgt vocabularies from train set to prevent data leakage
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
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.6f}")
    
    return total_loss / len(loader)


# ============================================================
# Validation
# ============================================================

def validate(model, loader, criterion, device):
    """
    Run validation. No gradient computation.
    Returns: average val loss
    """
    
    # Set model to evaluation mode
    model.eval()
    total_loss = 0.0

    # tqdm visualizer
    pbar = tqdm(loader, desc="Validation")
    
    # We don't need gradients for evaluation
    with torch.no_grad():
        # loop over batches, compute loss, accumulate
        for src, tgt in pbar:
            src, tgt = src.to(device), tgt.to(device)

            # Teacher forcing shift
            decoder_input = tgt[:, :-1]
            target = tgt[:, 1:]
            
            # Take model predictions
            outputs = model(src, decoder_input)
        
            # Calculate loss
            loss = criterion(
                outputs.reshape(-1, outputs.size(-1)),
                target.reshape(-1)
            )

            total_loss += loss.item()
            
    return total_loss / len(loader)


# ============================================================
# Sample translation (sanity check)
# ============================================================

def translate_sample(model, src_vocab, tgt_vocab, sentence, device, max_len=50):
    """
    Greedy decode a single sentence. For qualitative monitoring.
    """

    # Set the model to evaluation mode since we don't need gradients
    model.eval()

    # tokenize, encode, add <sos>/<eos>
    tokens = tokenize_en(sentence)
    src_ids = src_vocab.encode(tokens + [Vocabulary.EOS_TOKEN])
    tgt_ids = [tgt_vocab.sos_idx]
    src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)                        # shape: [1, src_len]

    # encoder forward
    with torch.no_grad():
        # Src mask and encoder output
        src_mask = make_src_mask(src, pad_idx=0)
        enc_out = model.encoder(src, src_mask)

        # autoregressive decoding loop
        for _ in range(max_len):
            # Target and target mask
            tgt = torch.tensor(tgt_ids, dtype=torch.long, device=device).unsqueeze(0)               # shape: [1, current_len]
            tgt_mask = make_tgt_mask(tgt, pad_idx=0)
            
            # Decode the next token which has highest logit
            dec_out = model.decoder(tgt, enc_out, src_mask, tgt_mask)                               # shape: [1, current_len, vocab]
            next_token = dec_out[0, -1].argmax().item()
            tgt_ids.append(next_token)
            
            # <eos> token: End of sentence
            if next_token == tgt_vocab.eos_idx: 
                break
    
    # decode IDs back to tokens with skipping <sos> and <eos> tokens
    # skip <sos>
    result_ids = tgt_ids[1:]
    
    # if <eos> exists, skip it also
    if result_ids and result_ids[-1] == tgt_vocab.eos_idx:
        result_ids = result_ids[:-1]
    tokens = tgt_vocab.decode(result_ids)
    
    return ' '.join(tokens)


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(model, optimizer, epoch, val_loss, path):
    """Save model, optimizer state, metadata."""
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, path)


# ============================================================
# Main
# ============================================================

def main():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Prepare data
    train_loader, val_loader, src_vocab, tgt_vocab = prepare_data(config)
    
    # Create model
    model = Transformer(
       src_vocab_size=len(src_vocab),
       tgt_vocab_size=len(tgt_vocab),
       d_model=config['d_model'],
       num_layers=config['num_layers'],
       num_heads=config['num_heads'],
       d_ff=config['d_ff'],
       max_len=config['max_len'],
       dropout=config['dropout'],
       pad_idx=config['pad_idx'],
    ).to(device)
    
    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(
       ignore_index=config['pad_idx'],
       label_smoothing=config['label_smoothing']
    )

    optimizer = Adam(
       model.parameters(),
       lr=1.0,  # scheduler will overwrite the learning rate
       betas=config['adam_betas'],
       eps=config['adam_eps']
    )
    
    scheduler = LambdaLR(
       optimizer,
       lr_lambda=lambda step: noam_schedule(max(step, 1), config['d_model'], config['warmup_steps'])
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    os.makedirs('checkpoints', exist_ok=True)
    
    for epoch in range(config['num_epochs']):
       train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, config['clip_grad'])
       val_loss = validate(model, val_loader, criterion, device)
       
       train_losses.append(train_loss)
       val_losses.append(val_loss)
       
       print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
       
       # Every 5 epoch, translate a sample
       if (epoch + 1) % 5 == 0:
           sample = translate_sample(model, src_vocab, tgt_vocab, "A man is walking in the park.", device)
           print(f"Sample: {sample}")
       
       # Checkpoint
       if val_loss < best_val_loss:
           best_val_loss = val_loss
           save_checkpoint(model, optimizer, epoch, val_loss, f'checkpoints/best_model_epoch{epoch+1}.pt')
    
    # Save loss curve
    os.makedirs('figures', exist_ok=True)
    plt.figure()
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('figures/loss_curve.png')
    

if __name__ == '__main__':
    main()