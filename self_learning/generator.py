"""
Self-Instruct Synthetic Generation Engine.
Orchestrates autoregressive model sampling over seed prompt pools to construct
diverse candidate task-response pairs for preference alignment.
"""

import torch
from typing import List, Dict
from tokenizer.bpe_tokenizer import BPETokenizer


class SelfInstructGenerator:
    """
    Generates synthetic task solutions for seed prompts using autoregressive sampling.
    """

    def __init__(self, model, tokenizer: BPETokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def generate_candidates_for_prompt(
        self,
        prompt: str,
        n_candidates: int = 4,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.9
    ) -> List[str]:
        """
        Generates N candidate completion attempts for a given prompt string.
        """
        formatted_prompt = f"Prompt: {prompt}\nSolution:"
        prompt_ids = self.tokenizer.encode(formatted_prompt, add_bos=True, add_eos=False)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        candidates = []
        for _ in range(n_candidates):
            gen_tensor = self.model.generate(
                prompt_ids=prompt_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                eos_id=self.tokenizer.eos_id
            )
            # Slice off prompt tokens to keep completion solution
            completion_ids = gen_tensor[0, len(prompt_ids):].tolist()
            solution_str = self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            candidates.append(solution_str)

        return candidates

    def generate_synthetic_batch(
        self,
        seed_prompts: List[str],
        candidates_per_prompt: int = 4,
        temperature: float = 0.8
    ) -> List[Dict[str, List[str]]]:
        """
        Generates candidate completions across a batch of seed prompts.
        
        Returns:
            List of dicts: [{"prompt": prompt, "candidates": [sol1, sol2, ...]}, ...]
        """
        results = []
        self.model.eval()

        for prompt in seed_prompts:
            candidates = self.generate_candidates_for_prompt(
                prompt=prompt,
                n_candidates=candidates_per_prompt,
                temperature=temperature
            )
            results.append({
                "prompt": prompt,
                "candidates": candidates
            })

        return results
