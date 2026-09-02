"""
Unit tests for individual subsystems of the Self-Learning Autoregressive LLM codebase.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenizer.bpe_tokenizer import BPETokenizer
from data.dataset import create_causal_mask


def test_tokenizer():
    print("Testing BPE Tokenizer...")
    text = "The quick brown fox jumps over the lazy dog."
    tokenizer = BPETokenizer()
    tokenizer.train(text, vocab_size=280)
    
    encoded = tokenizer.encode(text, add_bos=True, add_eos=True)
    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    
    assert len(encoded) > 0, "Encoding returned empty list"
    assert decoded.strip() == text.strip(), f"Roundtrip decoding failed: '{decoded}' != '{text}'"
    print("[OK] Tokenizer test passed!")


def test_causal_mask():
    print("Testing Causal Mask generator...")
    mask = create_causal_mask(seq_len=4)
    assert mask.shape == (4, 4), f"Unexpected mask shape: {mask.shape}"
    assert mask[0, 1].item() is True, "Causal mask should mask out future tokens (upper triangular True)"
    assert mask[1, 0].item() is False, "Causal mask should allow past tokens (lower triangular False)"
    print("[OK] Causal mask test passed!")


if __name__ == "__main__":
    test_tokenizer()
    test_causal_mask()
    print("All component tests passed cleanly!")
