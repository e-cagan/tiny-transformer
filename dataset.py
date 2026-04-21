"""
Module for creating and processing the dataset.
"""

import torch
import spacy
from collections import Counter
from torch.nn.utils.rnn import pad_sequence


# Load spacy models
en_model = spacy.load("en_core_web_sm")
de_model = spacy.load("de_core_news_sm")


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


if __name__ == '__main__':
    # Test out the vocabulary
    def dummy_tokenizer(s):
        return s.lower().split()

    v = Vocabulary()
    print("specials ok:", v.pad_idx == 0, v.sos_idx == 1, v.eos_idx == 2, v.unk_idx == 3)

    sentences = ["hello world", "hello cagan", "world peace"]
    v.build(sentences, dummy_tokenizer, min_freq=1)
    print("vocab size:", len(v))  # expected: 4 special + 4 unique (hello, world, cagan, peace) = 8

    ids = v.encode(["hello", "unknownword", "world"])
    print("encoded:", ids)  # expected: unknownword → unk_idx (3)

    tokens = v.decode(ids)
    print("decoded:", tokens)  # expected: ["hello", "<unk>", "world"]