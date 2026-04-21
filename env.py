"""
Module for setting up the workspace environment by installing the multi30k dataset and spacy en_core_web_sm, de_core_news_sm models.
"""

import sys
import random
import spacy
from datasets import load_dataset


if __name__ == '__main__':
    # Load the dataset
    ds = load_dataset("bentrevett/multi30k")

    # Create random index and check a random train pair with it
    for _ in range(5):
        idx = random.randint(0, len(ds['train']) - 1)
        # Print out 5 random pair
        print(f"Pair for index {idx}:")
        print(ds['train'][idx])
        print("="*50)

    # Download spacy models
    spacy.cli.download("en_core_web_sm")
    spacy.cli.download("de_core_news_sm")

    sys.exit(0)