"""
Byte-Pair Encoding (BPE) Tokenizer implementation from scratch.
Provides byte-level fallback, vocabulary training, token encoding/decoding,
and serialization/deserialization.
"""

import json
import os
from typing import Dict, List, Tuple, Set


class BPETokenizer:
    """Pure Python Byte-Pair Encoding Tokenizer with Byte-Level Fallback."""

    SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]

    def __init__(self, vocab: Dict[int, bytes] = None, merges: List[Tuple[bytes, bytes]] = None):
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3

        if vocab is not None and merges is not None:
            self.vocab = vocab
            self.merges = merges
            self.inverse_vocab = {v: k for k, v in self.vocab.items()}
            self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
        else:
            self.vocab: Dict[int, bytes] = {}
            self.inverse_vocab: Dict[bytes, int] = {}
            self.merges: List[Tuple[bytes, bytes]] = []
            self.merge_ranks: Dict[Tuple[bytes, bytes], int] = {}
            self._init_base_vocab()

    def _init_base_vocab(self):
        """Initializes special tokens and all 256 byte tokens."""
        for idx, token_str in enumerate(self.SPECIAL_TOKENS):
            token_bytes = token_str.encode("utf-8")
            self.vocab[idx] = token_bytes
            self.inverse_vocab[token_bytes] = idx

        offset = len(self.SPECIAL_TOKENS)
        for b in range(256):
            token_bytes = bytes([b])
            idx = offset + b
            self.vocab[idx] = token_bytes
            self.inverse_vocab[token_bytes] = idx

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        """Trains BPE merge rules on raw text up to targeted vocab_size."""
        assert vocab_size >= 256 + len(self.SPECIAL_TOKENS), "vocab_size must be >= 260"
        self._init_base_vocab()
        
        # Split text into lines to learn subword pairs instead of document-level merges
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        sequences = [[bytes([b]) for b in line.encode("utf-8")] for line in lines]

        num_merges = vocab_size - len(self.vocab)

        for i in range(num_merges):
            # Count pair frequencies
            pair_counts: Dict[Tuple[bytes, bytes], int] = {}
            for seq in sequences:
                for pair in zip(seq[:-1], seq[1:]):
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1

            if not pair_counts:
                break

            # Find most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[best_pair] < 2:
                # Stop if pairs appear less than twice
                break

            new_token_bytes = best_pair[0] + best_pair[1]
            if len(new_token_bytes) > 16:
                # Prevent merging tokens into overly long strings
                continue
            new_id = len(self.vocab)

            self.vocab[new_id] = new_token_bytes
            self.inverse_vocab[new_token_bytes] = new_id
            self.merges.append(best_pair)
            self.merge_ranks[best_pair] = i

            # Merge pair in sequences
            new_sequences = []
            for seq in sequences:
                new_seq = []
                j = 0
                while j < len(seq):
                    if j < len(seq) - 1 and (seq[j], seq[j + 1]) == best_pair:
                        new_seq.append(new_token_bytes)
                        j += 2
                    else:
                        new_seq.append(seq[j])
                        j += 1
                new_sequences.append(new_seq)
            sequences = new_sequences

            if verbose and (i + 1) % 500 == 0:
                print(f"Iter {i+1}/{num_merges}: Merged pair {best_pair} -> ID {new_id}")

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        """Encodes text into a list of token IDs using learned BPE merge rules."""
        if not text:
            tokens = []
            if add_bos:
                tokens.insert(0, self.bos_id)
            if add_eos:
                tokens.append(self.eos_id)
            return tokens

        text_bytes = text.encode("utf-8")
        seq = [bytes([b]) for b in text_bytes]

        while len(seq) >= 2:
            # Find the merge pair with the lowest rank (highest priority)
            pairs = zip(seq[:-1], seq[1:])
            min_pair = None
            min_rank = float("inf")

            for pair in pairs:
                rank = self.merge_ranks.get(pair, float("inf"))
                if rank < min_rank:
                    min_rank = rank
                    min_pair = pair

            if min_pair is None or min_rank == float("inf"):
                break

            # Apply merge
            new_seq = []
            j = 0
            while j < len(seq):
                if j < len(seq) - 1 and (seq[j], seq[j + 1]) == min_pair:
                    new_seq.append(min_pair[0] + min_pair[1])
                    j += 2
                else:
                    new_seq.append(seq[j])
                    j += 1
            seq = new_seq

        token_ids = [self.inverse_vocab.get(b, self.unk_id) for b in seq]

        if add_bos:
            token_ids.insert(0, self.bos_id)
        if add_eos:
            token_ids.append(self.eos_id)

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decodes token IDs back into UTF-8 text."""
        byte_chunks = []
        for tid in token_ids:
            if skip_special_tokens and tid in (self.pad_id, self.unk_id, self.bos_id, self.eos_id):
                continue
            chunk = self.vocab.get(tid, b"")
            byte_chunks.append(chunk)

        full_bytes = b"".join(byte_chunks)
        return full_bytes.decode("utf-8", errors="replace")

    def save(self, filepath: str):
        """Serializes vocabulary and merge rules to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "vocab": {str(k): v.decode("latin-1") for k, v in self.vocab.items()},  # store bytes safely
            "merges": [[p[0].decode("latin-1"), p[1].decode("latin-1")] for p in self.merges]
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "BPETokenizer":
        """Deserializes tokenizer from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        vocab = {int(k): v.encode("latin1") for k, v in data["vocab"].items()}
        merges = [(p[0].encode("latin1"), p[1].encode("latin1")) for p in data["merges"]]
        return cls(vocab=vocab, merges=merges)

    def get_vocab_size(self) -> int:
        return len(self.vocab)
