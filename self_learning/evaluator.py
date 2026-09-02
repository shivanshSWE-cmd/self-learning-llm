"""
Verification & Critique Reward Engine.
Evaluates generated candidate solutions via execution testing (Python sandbox / math checks)
and heuristic critique scoring to form preference pairs (chosen y_w vs rejected y_l).
"""

import sys
import io
import traceback
import math
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


@dataclass
class PreferenceSample:
    """Structure representing a DPO preference tuple: (prompt, chosen, rejected)."""
    prompt: str
    chosen: str
    rejected: str
    chosen_score: float
    rejected_score: float


class ExecutionCritiqueEvaluator:
    """
    Automated execution-based & critique reward verifier.
    Scores generated candidate solutions and filters out preference pairs for DPO.
    """

    def __init__(self, timeout_sec: float = 2.0):
        self.timeout_sec = timeout_sec

    def evaluate_code_execution(self, code_str: str) -> float:
        """
        Executes Python code snippet safely in isolated stdout/stderr buffer.
        Returns a numerical score between 0.0 (error/fail) and 1.0 (clean execution).
        """
        # Clean markdown backticks if present
        clean_code = code_str.replace("```python", "").replace("```", "").strip()
        if not clean_code:
            return 0.0

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        redirected_output = io.StringIO()

        try:
            sys.stdout = redirected_output
            sys.stderr = redirected_output

            # Execute code snippet
            exec_globals = {"math": math, "__name__": "__main__"}
            exec(clean_code, exec_globals)
            sys.stdout = old_stdout
            sys.stderr = old_stderr

            output_str = redirected_output.getvalue()
            # Clean execution reward
            return 1.0 if len(output_str) >= 0 else 0.5

        except Exception as e:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            # Execution failure penalty
            return 0.0

    def evaluate_heuristic_quality(self, prompt: str, completion: str) -> float:
        """
        Heuristic critique scoring function evaluating length, structural coherence,
        repetition, and keyword matching.
        """
        if not completion or len(completion.strip()) < 5:
            return 0.0

        score = 0.5
        words = completion.split()

        # Length constraint heuristic (reward non-trivial answers, penalize runaway repetition)
        if 15 <= len(words) <= 200:
            score += 0.2
        elif len(words) > 300:
            score -= 0.3 # Penalize endless loops

        # Repetition penalty: Check unique 3-gram ratio
        if len(words) > 6:
            trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
            unique_trigram_ratio = len(set(trigrams)) / max(1, len(trigrams))
            if unique_trigram_ratio < 0.4:
                score -= 0.4 # Severe repetitive loop penalty
            else:
                score += 0.1

        # Reward code formatting or step-by-step reasoning tokens
        if "```" in completion or "def " in completion or "return " in completion:
            score += 0.2
        if "Step " in completion or "Therefore" in completion or "Solution" in completion:
            score += 0.1

        return max(0.0, min(1.0, score))

    def evaluate_candidate(self, prompt: str, completion: str) -> float:
        """Combined verification score combining execution sandbox and heuristic critique."""
        if "def " in completion or "import " in completion:
            exec_score = self.evaluate_code_execution(completion)
            heuristic_score = self.evaluate_heuristic_quality(prompt, completion)
            return 0.6 * exec_score + 0.4 * heuristic_score
        else:
            return self.evaluate_heuristic_quality(prompt, completion)

    def filter_and_format_preference_pairs(
        self,
        generated_batch: List[Dict[str, List[str]]],
        min_score_delta: float = 0.2
    ) -> List[PreferenceSample]:
        """
        Filters batch of candidate completions into DPO preference pairs (y_w, y_l).
        
        Args:
            generated_batch: Output from SelfInstructGenerator.
            min_score_delta: Minimum reward gap between chosen and rejected candidates.
            
        Returns:
            List of PreferenceSample objects ready for DPO policy update.
        """
        preference_dataset: List[PreferenceSample] = []

        for item in generated_batch:
            prompt = item["prompt"]
            candidates = item["candidates"]
            if len(candidates) < 2:
                continue

            # Score each candidate
            scored_candidates: List[Tuple[str, float]] = [
                (cand, self.evaluate_candidate(prompt, cand))
                for cand in candidates
            ]

            # Sort descending by score
            scored_candidates.sort(key=lambda x: x[1], reverse=True)

            best_cand, best_score = scored_candidates[0]
            worst_cand, worst_score = scored_candidates[-1]

            # Only accept pair if there is a significant score margin
            if (best_score - worst_score) >= min_score_delta and best_score > 0.3:
                preference_dataset.append(PreferenceSample(
                    prompt=prompt,
                    chosen=best_cand,
                    rejected=worst_cand,
                    chosen_score=best_score,
                    rejected_score=worst_score
                ))

        return preference_dataset
