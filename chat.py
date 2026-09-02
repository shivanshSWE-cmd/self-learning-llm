"""
Interactive Chat & Evaluation CLI for the Self-Learning Autoregressive LLM.
Allows testing model checkpoints interactively or comparing outputs between base and self-improved models.
"""

import os
import sys
import argparse
import torch

from config import ModelConfig
from tokenizer.bpe_tokenizer import BPETokenizer
from models.transformer import TransformerLM


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", ModelConfig())
    
    model = TransformerLM(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config


def interactive_chat(checkpoint_path: str = "checkpoints/self_improved_cycle_2.pt"):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_path = "tokenizer/vocab.json"

    if not os.path.exists(tokenizer_path):
        print(f"Error: Tokenizer vocabulary not found at {tokenizer_path}. Run main.py first.")
        return

    tokenizer = BPETokenizer.load(tokenizer_path)
    print(f"Loading model checkpoint: {checkpoint_path}...")
    model, config = load_model_from_checkpoint(checkpoint_path, device)
    print(f"Model loaded successfully! ({sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters)")

    print("\n=======================================================")
    print("      Interactive LLM Tester & Chat Environment        ")
    print("=======================================================")
    print("Commands:")
    print("  'exit' or 'quit' - Exit the session")
    print("  'temp <float>'   - Adjust temperature (default: 0.7)")
    print("  'tokens <int>'   - Adjust max output tokens (default: 64)")
    print("=======================================================\n")

    temperature = 0.7
    top_p = 0.9
    max_new_tokens = 64

    while True:
        try:
            prompt = input("\n[USER Prompt] > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat session.")
            break

        if not prompt:
            continue
        if prompt.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if prompt.startswith("temp "):
            try:
                temperature = float(prompt.split()[1])
                print(f"Set sampling temperature to {temperature}")
            except Exception:
                print("Invalid temperature value.")
            continue
        if prompt.startswith("tokens "):
            try:
                max_new_tokens = int(prompt.split()[1])
                print(f"Set max new tokens to {max_new_tokens}")
            except Exception:
                print("Invalid max tokens value.")
            continue

        formatted_prompt = f"Prompt: {prompt}\nSolution:"
        prompt_ids = tokenizer.encode(formatted_prompt, add_bos=True)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        print("\n[LLM Generating...] ", end="", flush=True)
        generated_tensor = model.generate(
            prompt_ids=prompt_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_id=tokenizer.eos_id
        )

        completion_ids = generated_tensor[0, len(prompt_ids):].tolist()
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

        print("\n-------------------------------------------------------")
        print(completion_text if completion_text else "[Empty generation]")
        print("-------------------------------------------------------")


def compare_models(base_ckpt: str = "checkpoints/base_model.pt", improved_ckpt: str = "checkpoints/self_improved_cycle_2.pt"):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BPETokenizer.load("tokenizer/vocab.json")

    print(f"Loading Base Model ({base_ckpt})...")
    base_model, _ = load_model_from_checkpoint(base_ckpt, device)

    print(f"Loading Self-Improved Model ({improved_ckpt})...")
    improved_model, _ = load_model_from_checkpoint(improved_ckpt, device)

    test_prompts = [
        "Write a Python function to add two numbers.",
        "Explain autoregressive language models.",
        "Write a function to check if a number is even.",
        "Calculate the factorial of a number."
    ]

    print("\n==========================================================================")
    print("              MODEL COMPARISON: BASE PRE-TRAINED vs SELF-IMPROVED         ")
    print("==========================================================================\n")

    for prompt in test_prompts:
        formatted = f"Prompt: {prompt}\nSolution:"
        prompt_ids = tokenizer.encode(formatted, add_bos=True)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        base_gen = base_model.generate(prompt_tensor, max_new_tokens=48, temperature=0.7)
        improved_gen = improved_model.generate(prompt_tensor, max_new_tokens=48, temperature=0.7)

        base_text = tokenizer.decode(base_gen[0, len(prompt_ids):].tolist(), skip_special_tokens=True).strip()
        improved_text = tokenizer.decode(improved_gen[0, len(prompt_ids):].tolist(), skip_special_tokens=True).strip()

        print(f"PROMPT: {prompt}")
        print(f"  [Base Model]        : {base_text}")
        print(f"  [Self-Improved Model]: {improved_text}")
        print("-" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test & Chat CLI for Self-Learning LLM")
    parser.add_argument("--ckpt", type=str, default="checkpoints/self_improved_cycle_2.pt", help="Path to checkpoint")
    parser.add_argument("--compare", action="store_true", help="Compare base model vs self-improved model")
    args = parser.parse_args()

    if args.compare:
        compare_models()
    else:
        interactive_chat(checkpoint_path=args.ckpt)
