"""
Script to train the BPE Tokenizer on a raw text corpus and serialize to JSON vocabulary file.
"""

import sys
import os

# Add parent path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenizer.bpe_tokenizer import BPETokenizer


def train_and_save_tokenizer(
    corpus_file: str = "data/corpus.txt",
    output_file: str = "tokenizer/vocab.json",
    vocab_size: int = 1000
):
    print(f"Loading corpus from: {corpus_file}")
    if not os.path.exists(corpus_file):
        print(f"Creating sample corpus at: {corpus_file}")
        os.makedirs(os.path.dirname(corpus_file), exist_ok=True)
        sample_text = (
            "The quick brown fox jumps over the lazy dog.\n"
            "Autoregressive language models predict the next token given preceding context.\n"
            "Self-learning LLMs generate synthetic data, evaluate candidate solutions, and perform DPO preference updates.\n"
            "Deep Learning Systems Engineering requires pure PyTorch implementations of RoPE, SwiGLU, RMSNorm, and GQA.\n"
        ) * 100
        with open(corpus_file, "w", encoding="utf-8") as f:
            f.write(sample_text)

    with open(corpus_file, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Training BPE Tokenizer with target vocab_size={vocab_size} on {len(text)} characters...")
    tokenizer = BPETokenizer()
    tokenizer.train(text, vocab_size=vocab_size, verbose=True)

    tokenizer.save(output_file)
    print(f"Successfully trained and saved vocabulary ({tokenizer.get_vocab_size()} tokens) to {output_file}")


if __name__ == "__main__":
    train_and_save_tokenizer()
