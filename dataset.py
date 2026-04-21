"""
Module for creating and processing the dataset.
"""

import torch
import spacy
from collections import Counter
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence


# Load spacy models
en_model = spacy.load("en_core_web_sm")
de_model = spacy.load("de_core_news_sm")


# Tokenizer helper functions
def tokenize_en(text):
    tokens = [tok.text for tok in en_model(text)]
    return tokens


def tokenize_de(text):
    tokens = [tok.text for tok in de_model(text)]
    return tokens


# Build vocabulary class to create vocabulary
class Vocabulary:
    """
    Vocabulary class
    """

    # Special tokens
    PAD_TOKEN = "<pad>"  # id 0
    SOS_TOKEN = "<sos>"  # id 1
    EOS_TOKEN = "<eos>"  # id 2
    UNK_TOKEN = "<unk>"  # id 3
    
    def __init__(self, specials=None):
        # int -> str, str -> int
        self.itos = list()
        self.stoi = dict()
        
        # specials default: [PAD, SOS, EOS, UNK]
        if specials is None:
            specials = [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]
        
        for special in specials:
            self._add_token(special)
    
    def _add_token(self, token):
        self.itos.append(token)
        self.stoi[token] = len(self.itos) - 1

    def build(self, sentences, tokenizer, min_freq=2):
        # Tokenize the sentences and count freqs by each token 
        counter = Counter()
        tokens = [tokenizer(s) for s in sentences]
        for sentence_tokens in tokens:
            counter.update(sentence_tokens)
        
        # Add tokens based on min_freq threshold
        for token, freq in counter.items():
            if freq >= min_freq and token not in self.stoi:
                self._add_token(token)
    
    def encode(self, tokens):
        # token list -> ID list
        id_list = [self.stoi.get(token, self.unk_idx) for token in tokens]
        
        return id_list
    
    def decode(self, ids):
        # ID list -> token list
        token_list = [self.itos[i] for i in ids]
        
        return token_list
    
    def __len__(self):
        return len(self.itos)
    
    @property
    def pad_idx(self): 
        return self.stoi[self.PAD_TOKEN]

    @property
    def sos_idx(self): 
        return self.stoi[self.SOS_TOKEN]
    
    @property
    def eos_idx(self): 
        return self.stoi[self.EOS_TOKEN]
    
    @property
    def unk_idx(self): 
        return self.stoi[self.UNK_TOKEN]
    

# Build a collate function to apply padding towards sentences
class Collate:
    """
    Collate function class
    """

    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        src_list = list()
        target_list = list()

        for src_item, target_item in batch:
            src_list.append(src_item)
            target_list.append(target_item)

        # Apply padding
        # batch_first=True ==> Shape of (Batch_size, Max_Seq_Len)
        src_padded = pad_sequence(sequences=src_list, batch_first=True, padding_value=self.pad_idx)
        target_padded = pad_sequence(sequences=target_list, batch_first=True, padding_value=self.pad_idx)

        return src_padded, target_padded
    

# Pytorch dataset wrapper class
class Multi30kDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper class
    """

    def __init__(self, hf_split, src_tokenizer, tgt_tokenizer, src_vocab, tgt_vocab, max_len=None):
        super().__init__()
        self.hf_split = hf_split
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.hf_split)
    
    def __getitem__(self, index):
        # Take the split dict to tokenize sentences for src and tgt
        split_dict = self.hf_split[index]

        # Tokenize english and german sentences
        eng_tokens = self.src_tokenizer(split_dict['en'])
        ger_tokens = self.tgt_tokenizer(split_dict['de'])

        # Arrange some free slots to tokens if there is a max length
        if self.max_len is not None:
            eng_tokens = eng_tokens[:self.max_len - 2]
            ger_tokens = ger_tokens[:self.max_len - 2]

        # Add the special tokens
        src_tokens_with_special = eng_tokens + [Vocabulary.EOS_TOKEN]
        tgt_tokens_with_special = [Vocabulary.SOS_TOKEN] + ger_tokens + [Vocabulary.EOS_TOKEN]

        # Encode tokens to IDs
        src_ids = self.src_vocab.encode(src_tokens_with_special)
        tgt_ids = self.tgt_vocab.encode(tgt_tokens_with_special)

        # Convert IDs to tensors for dataloader
        src_tensor = torch.tensor(data=src_ids, dtype=torch.long)
        tgt_tensor = torch.tensor(data=tgt_ids, dtype=torch.long)

        return src_tensor, tgt_tensor


if __name__ == '__main__':
    # Test out the dataset pipeline
    ds = load_dataset("bentrevett/multi30k")
    train = ds['train']

    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()

    # Build vocab from train set
    src_vocab.build([ex['en'] for ex in train], tokenize_en, min_freq=2)
    tgt_vocab.build([ex['de'] for ex in train], tokenize_de, min_freq=2)

    print(f"src vocab size: {len(src_vocab)}")  # ~5000-6000
    print(f"tgt vocab size: {len(tgt_vocab)}")  # ~7000-8000

    dataset = Multi30kDataset(train, tokenize_en, tokenize_de, src_vocab, tgt_vocab, max_len=50)
    src, tgt = dataset[0]
    print("src:", src)
    print("tgt:", tgt)
    print("decoded src:", src_vocab.decode(src.tolist()))
    print("decoded tgt:", tgt_vocab.decode(tgt.tolist()))

    collate_fn = Collate(pad_idx=src_vocab.pad_idx)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    src_batch, tgt_batch = next(iter(loader))
    print("src_batch shape:", src_batch.shape)  # (4, max_len_in_batch)
    print("tgt_batch shape:", tgt_batch.shape)
    print("src_batch:\n", src_batch)
    print("tgt_batch:\n", tgt_batch)