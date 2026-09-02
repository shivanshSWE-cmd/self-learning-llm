"""
Custom Loss Functions implemented in pure PyTorch:
1. Autoregressive Causal Cross-Entropy Loss
2. Direct Preference Optimization (DPO) Loss
3. Elastic Weight Consolidation (EWC) Loss & Fisher Information Estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


def compute_causal_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100
) -> torch.Tensor:
    """
    Computes Autoregressive Causal Cross-Entropy Loss.
    
    Args:
        logits: Output logits of shape [batch_size, seq_len, vocab_size]
        targets: Target token IDs of shape [batch_size, seq_len]
        ignore_index: Token ID index to ignore in loss (padding mask)
    """
    # Flatten tensors: [batch_size * seq_len, vocab_size] vs [batch_size * seq_len]
    shift_logits = logits.view(-1, logits.size(-1))
    shift_targets = targets.view(-1)
    
    loss = F.cross_entropy(shift_logits, shift_targets, ignore_index=ignore_index)
    return loss


def _get_batch_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_pad_token_id: int = -100
) -> torch.Tensor:
    """
    Computes sequence log probabilities for response tokens.
    
    Args:
        logits: [batch_size, seq_len, vocab_size]
        labels: [batch_size, seq_len] with response tokens and -100 for prompt/padding
    """
    # Compute per-token log softmax
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Target shift: predict labels[t] from logits[t-1]
    # For alignment, logits[:, :-1] predicts labels[:, 1:]
    shift_logits = log_probs[:, :-1, :]
    shift_labels = labels[:, 1:].clone()

    loss_mask = shift_labels != label_pad_token_id
    shift_labels[~loss_mask] = 0 # Dummy replace for gather indexing

    # Gather log prob of actual target token
    per_token_log_probs = torch.gather(shift_logits, dim=2, index=shift_labels.unsqueeze(2)).squeeze(2)

    # Mask out prompt and padding positions
    return (per_token_log_probs * loss_mask).sum(dim=-1)


def compute_dpo_loss(
    policy_chosen_logits: torch.Tensor,
    policy_rejected_logits: torch.Tensor,
    ref_chosen_logits: torch.Tensor,
    ref_rejected_logits: torch.Tensor,
    chosen_labels: torch.Tensor,
    rejected_labels: torch.Tensor,
    beta: float = 0.1,
    label_pad_token_id: int = -100
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes Direct Preference Optimization (DPO) Loss.
    
    Math Formulation:
    L_DPO = - E_{(x, y_w, y_l)} [ log sigmoid ( beta * ( log pi_theta(y_w|x) - log pi_ref(y_w|x) )
                                               - beta * ( log pi_theta(y_l|x) - log pi_ref(y_l|x) ) ) ]
                                               
    Returns:
        dpo_loss, chosen_rewards, rejected_rewards
    """
    # Calculate log probs under Policy model pi_theta
    policy_chosen_logps = _get_batch_log_probs(policy_chosen_logits, chosen_labels, label_pad_token_id)
    policy_rejected_logps = _get_batch_log_probs(policy_rejected_logits, rejected_labels, label_pad_token_id)

    # Calculate log probs under Reference model pi_ref (frozen pre-trained baseline)
    with torch.no_grad():
        ref_chosen_logps = _get_batch_log_probs(ref_chosen_logits, chosen_labels, label_pad_token_id)
        ref_rejected_logps = _get_batch_log_probs(ref_rejected_logits, rejected_labels, label_pad_token_id)

    # Compute log probability ratios
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps

    logits = beta * (pi_logratios - ref_logratios)

    # Negative log-sigmoid DPO loss
    loss = -F.logsigmoid(logits).mean()

    # Implicit reward metrics for tracking alignment progress
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()

    return loss, chosen_rewards, rejected_rewards


def estimate_fisher_information(
    model: nn.Module,
    dataloader,
    num_samples: int = 256,
    device: torch.device = None
) -> Dict[str, torch.Tensor]:
    """
    Estimates the diagonal elements of the Empirical Fisher Information Matrix (FIM)
    over baseline pre-training replay data.
    Used for Elastic Weight Consolidation (EWC) to prevent catastrophic forgetting.
    """
    model.eval()
    fisher_dict = {name: torch.zeros_like(param) for name, param in model.named_parameters() if param.requires_grad}

    samples_processed = 0
    for x, y in dataloader:
        if samples_processed >= num_samples:
            break

        if device is not None:
            x, y = x.to(device), y.to(device)

        model.zero_grad()
        logits = model(x)
        loss = compute_causal_ce_loss(logits, y)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # Accumulate squared gradients E[(d L / d theta)^2]
                fisher_dict[name] += param.grad.data.pow(2) * x.size(0)

        samples_processed += x.size(0)

    # Normalize by total samples evaluated
    for name in fisher_dict:
        fisher_dict[name] /= max(1, samples_processed)

    return fisher_dict


def compute_ewc_penalty(
    model: nn.Module,
    anchor_params: Dict[str, torch.Tensor],
    fisher_matrix: Dict[str, torch.Tensor],
    ewc_lambda: float = 500.0
) -> torch.Tensor:
    """
    Computes Elastic Weight Consolidation (EWC) Penalty Loss:
    L_EWC = (lambda / 2) * sum_i F_i * (theta_i - theta_anchor_i)^2
    """
    ewc_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if name in fisher_matrix and name in anchor_params:
            fisher = fisher_matrix[name]
            anchor = anchor_params[name]
            # Quadratic distance weighted by diagonal Fisher Information
            ewc_loss += (fisher * (param - anchor).pow(2)).sum()

    return (ewc_lambda / 2.0) * ewc_loss
