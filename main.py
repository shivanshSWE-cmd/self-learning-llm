"""
Autonomous Self-Improvement Driver Orchestrator.
Executes the closed-loop self-learning LLM pipeline:
1. Base Pre-training Warmup
2. Reference Model & Replay Buffer Snapshot (Fisher Matrix for EWC)
3. Self-Instruct Synthetic Generation over Seed Prompt Pool
4. Execution & Critique Verification -> DPO Preference Pair Filtering (y_w vs y_l)
5. Policy Update using DPO Loss + EWC Catastrophic Forgetting Penalty
6. Model Validation & Checkpoint Serialization
"""

import os
import sys
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    ModelConfig, DataConfig, TrainConfig,
    DPOConfig, EWCConfig, SelfLearningConfig
)
from tokenizer.bpe_tokenizer import BPETokenizer
from data.dataset import text_to_memmap, create_dataloader
from models.transformer import TransformerLM
from engine.trainer import DistributedTrainer
from engine.losses import compute_dpo_loss, compute_ewc_penalty
from self_learning.generator import SelfInstructGenerator
from self_learning.evaluator import ExecutionCritiqueEvaluator
from self_learning.replay_buffer import BaselineReplayBuffer


def run_self_improvement_pipeline():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("==========================================================================")
    print("       Starting Self-Learning Autoregressive LLM System (Pure PyTorch)     ")
    print("==========================================================================")

    # Initialize Hardware Device & Seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Set PyTorch Float32 Matmul Precision
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    # --------------------------------------------------------------------------
    # Stage 1: Configurations & Data Pipeline Setup
    # --------------------------------------------------------------------------
    model_cfg = ModelConfig(
        vocab_size=1000,
        dim=256,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,  # Grouped-Query Attention (GQA ratio 2:1)
        max_seq_len=256
    )
    data_cfg = DataConfig(seq_len=16, batch_size=4)
    train_cfg = TrainConfig(max_steps=10, warmup_steps=2, eval_interval=2, use_amp=False)
    dpo_cfg = DPOConfig(beta=0.1, learning_rate=1e-5)
    ewc_cfg = EWCConfig(ewc_lambda=100.0, num_fisher_samples=16)
    self_cfg = SelfLearningConfig(num_cycles=2, prompts_per_cycle=4, candidates_per_prompt=4)

    # Prepare Sample Corpus & Train Tokenizer
    corpus_dir = "data"
    os.makedirs(corpus_dir, exist_ok=True)
    corpus_file = os.path.join(corpus_dir, "corpus.txt")

    sample_corpus = (
        "Write a Python function to add two numbers.\n"
        "def add(a, b):\n    return a + b\n\n"
        "Explain autoregressive language models.\n"
        "Autoregressive language models predict next tokens given previous context step by step.\n\n"
        "Write a Python code snippet to check if a number is even.\n"
        "def is_even(n):\n    return n % 2 == 0\n\n"
    ) * 300
    with open(corpus_file, "w", encoding="utf-8") as f:
        f.write(sample_corpus)

    tokenizer_path = "tokenizer/vocab.json"
    print("\n--- Training BPE Tokenizer ---")
    tokenizer = BPETokenizer()
    with open(corpus_file, "r", encoding="utf-8") as f:
        corpus_text = f.read()
    tokenizer.train(corpus_text, vocab_size=model_cfg.vocab_size)
    tokenizer.save(tokenizer_path)

    # Encode text corpus to memory-mapped binary dataset
    bin_file_path = os.path.join(corpus_dir, "dataset.bin")
    total_tokens = text_to_memmap(corpus_text, tokenizer, bin_file_path)
    print(f"Dataset memory-mapped to {bin_file_path} ({total_tokens} tokens)")

    dataloader = create_dataloader(bin_file_path, seq_len=data_cfg.seq_len, batch_size=data_cfg.batch_size)

    # --------------------------------------------------------------------------
    # Stage 2: TransformerLM Model Construction
    # --------------------------------------------------------------------------
    print("\n--- Building TransformerLM Decoder Model (Pure PyTorch) ---")
    model = TransformerLM(model_cfg)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {total_params / 1e6:.3f}M")

    # --------------------------------------------------------------------------
    # Stage 3: Base Pre-training Warmup
    # --------------------------------------------------------------------------
    print("\n--- Stage 1: Base Model Pre-Training Warmup ---")
    trainer = DistributedTrainer(model, train_cfg, device=device, is_distributed=False)
    trainer.train_epoch(dataloader)
    trainer.save_checkpoint("checkpoints/base_model.pt", meta={"stage": "pretraining"})

    # --------------------------------------------------------------------------
    # Stage 4: Snapshot Reference Model & Compute EWC Fisher Matrix
    # --------------------------------------------------------------------------
    print("\n--- Stage 2: Safeguard Setup (Snapshot Anchor Model & Compute EWC Fisher Matrix) ---")
    # Reference policy model pi_ref (frozen)
    ref_model = copy.deepcopy(model).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # Baseline Replay Buffer
    replay_buffer = BaselineReplayBuffer(capacity=200)
    replay_buffer.populate_from_dataloader(dataloader, max_samples=64)
    replay_buffer.snapshot_anchor_model(model)
    fisher_matrix = replay_buffer.compute_fisher_matrix(model, num_samples=ewc_cfg.num_fisher_samples, device=device)
    print(f"Computed Fisher Information Matrix across {len(fisher_matrix)} parameter sets.")

    # --------------------------------------------------------------------------
    # Stage 5: Closed-Loop Autonomous Self-Improvement (Generation -> Eval -> DPO + EWC)
    # --------------------------------------------------------------------------
    print("\n--- Stage 3: Autonomous Closed-Loop Self-Improvement ---")
    generator = SelfInstructGenerator(model=model, tokenizer=tokenizer, device=device)
    evaluator = ExecutionCritiqueEvaluator()

    seed_prompts = [
        "Write a Python function to return the square of a number.",
        "Write a function to check if a string is a palindrome.",
        "Calculate the factorial of a positive integer.",
        "Check if a list contains duplicate elements."
    ]

    for cycle in range(1, self_cfg.num_cycles + 1):
        print(f"\n=======================================================")
        print(f"   Self-Improvement Cycle {cycle}/{self_cfg.num_cycles}")
        print(f"=======================================================")

        # 1. Synthetic Candidate Generation
        print("1. Generating synthetic solutions for seed prompts...")
        gen_batch = generator.generate_synthetic_batch(
            seed_prompts=seed_prompts,
            candidates_per_prompt=self_cfg.candidates_per_prompt,
            temperature=self_cfg.temperature
        )

        # 2. Automated Execution & Critique Verification -> Preference Filtering
        print("2. Verifying & filtering candidates into preference pairs (y_w vs y_l)...")
        preference_pairs = evaluator.filter_and_format_preference_pairs(gen_batch, min_score_delta=0.1)
        print(f"Successfully generated {len(preference_pairs)} DPO preference samples.")

        if not preference_pairs:
            print("No preference pairs met threshold this cycle. Skipping DPO update.")
            continue

        # 3. Direct Preference Optimization (DPO) Policy Update with EWC Safeguard
        print("3. Executing DPO Policy Optimization with EWC penalty...")
        dpo_optimizer = optim.AdamW(model.parameters(), lr=dpo_cfg.learning_rate)
        model.train()

        for epoch in range(self_cfg.dpo_epochs_per_cycle):
            epoch_dpo_loss = 0.0
            epoch_ewc_loss = 0.0

            for sample in preference_pairs:
                dpo_optimizer.zero_grad()

                # Format prompt and response tokens
                prompt_ids = tokenizer.encode(f"Prompt: {sample.prompt}\nSolution:", add_bos=True)
                chosen_resp_ids = tokenizer.encode(sample.chosen, add_bos=False, add_eos=True)
                rejected_resp_ids = tokenizer.encode(sample.rejected, add_bos=False, add_eos=True)

                chosen_full = torch.tensor([prompt_ids + chosen_resp_ids], dtype=torch.long, device=device)
                rejected_full = torch.tensor([prompt_ids + rejected_resp_ids], dtype=torch.long, device=device)

                chosen_labels = chosen_full.clone()
                chosen_labels[:, :len(prompt_ids)] = -100 # Mask out prompt tokens

                rejected_labels = rejected_full.clone()
                rejected_labels[:, :len(prompt_ids)] = -100

                # Policy model pi_theta forward passes
                policy_chosen_logits = model(chosen_full)
                policy_rejected_logits = model(rejected_full)

                # Reference model pi_ref forward passes
                with torch.no_grad():
                    ref_chosen_logits = ref_model(chosen_full)
                    ref_rejected_logits = ref_model(rejected_full)

                # Compute DPO Loss
                dpo_loss, c_rew, r_rew = compute_dpo_loss(
                    policy_chosen_logits=policy_chosen_logits,
                    policy_rejected_logits=policy_rejected_logits,
                    ref_chosen_logits=ref_chosen_logits,
                    ref_rejected_logits=ref_rejected_logits,
                    chosen_labels=chosen_labels,
                    rejected_labels=rejected_labels,
                    beta=dpo_cfg.beta
                )

                # Compute EWC Catastrophic Forgetting Penalty Loss
                ewc_penalty = compute_ewc_penalty(
                    model=model,
                    anchor_params=replay_buffer.anchor_parameters,
                    fisher_matrix=fisher_matrix,
                    ewc_lambda=ewc_cfg.ewc_lambda
                )

                total_loss = dpo_loss + ewc_penalty
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                dpo_optimizer.step()

                epoch_dpo_loss += dpo_loss.item()
                epoch_ewc_loss += ewc_penalty.item()

            print(f"Cycle {cycle} | Epoch {epoch+1}/{self_cfg.dpo_epochs_per_cycle} | DPO Loss: {epoch_dpo_loss/len(preference_pairs):.4f} | EWC Loss: {epoch_ewc_loss/len(preference_pairs):.4f}")

        # Checkpoint updated model
        ckpt_path = f"checkpoints/self_improved_cycle_{cycle}.pt"
        trainer.save_checkpoint(ckpt_path, meta={"stage": f"self_improved_cycle_{cycle}"})

    # --------------------------------------------------------------------------
    # Stage 6: Final Generation Test Demonstration
    # --------------------------------------------------------------------------
    print("\n=======================================================")
    print("   Final Demonstration: Autoregressive Inference Generation")
    print("=======================================================")
    test_prompt = "Write a Python function to add two numbers."
    print(f"Test Prompt: '{test_prompt}'")
    
    prompt_ids = tokenizer.encode(f"Prompt: {test_prompt}\nSolution:", add_bos=True)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    
    generated_tensor = model.generate(prompt_tensor, max_new_tokens=64, temperature=0.7)
    completion = tokenizer.decode(generated_tensor[0].tolist(), skip_special_tokens=True)
    print("\nModel Output:")
    print(completion)
    print("\nSelf-Learning LLM Pipeline Execution Completed Successfully!")


if __name__ == "__main__":
    run_self_improvement_pipeline()
