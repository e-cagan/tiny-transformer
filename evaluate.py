"""
Evaluate trained Transformer on Multi30k test set with BLEU.
"""

import torch
import sacrebleu
from datasets import load_dataset
from tqdm import tqdm

from dataset import Vocabulary, tokenize_en, tokenize_de
from model.transformer import Transformer
from train import config, translate_sample
from translate import build_vocabs, load_model


def evaluate_bleu(model, src_vocab, tgt_vocab, test_set, device):
    """
    Translate all test sentences and compute corpus BLEU.
    Returns: BLEU object and list of (source, prediction, reference) tuples.
    """
    
    predictions = []
    references = []
    samples = []
    
    for ex in tqdm(test_set, desc="Translating test set"):
        pred = translate_sample(model, src_vocab, tgt_vocab, ex['en'], device)
        predictions.append(pred)
        references.append(ex['de'])
        samples.append((ex['en'], pred, ex['de']))
    
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    
    return bleu, samples


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Vocab + model
    src_vocab, tgt_vocab = build_vocabs()
    model = load_model('checkpoints/best_model_epoch18.pt', len(src_vocab), len(tgt_vocab), device)
    
    # Test set
    ds = load_dataset("bentrevett/multi30k")
    test = ds['test']
    
    # Evaluate
    bleu, samples = evaluate_bleu(model, src_vocab, tgt_vocab, test, device)
    
    print(f"\nBLEU score: {bleu.score:.2f}")
    print(f"BLEU details: {bleu}")
    
    # Print 5 examples
    print("\n=== Sample translations ===")
    for i, (src, pred, ref) in enumerate(samples[:5]):
        print(f"\n[{i+1}]")
        print(f"  EN:   {src}")
        print(f"  Pred: {pred}")
        print(f"  Ref:  {ref}")


if __name__ == '__main__':
    main()