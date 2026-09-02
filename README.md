# Self-Learning Autoregressive Large Language Model (LLM)

A production-grade, end-to-end implementation of a self-improving autoregressive Large Language Model (LLM) built from scratch in pure PyTorch (without high-level model wrappers such as Hugging Face `AutoModel`).

The architecture implements a closed autonomous self-improvement loop:
**Pre-training Warmup $\rightarrow$ Synthetic Self-Instruct Generation $\rightarrow$ Execution & Critique Verification $\rightarrow$ DPO Preference Alignment $\rightarrow$ EWC Continual Learning Safeguard $\rightarrow$ Checkpoint Serialization**.

---

## Key Features & Modular Structure

1. **Tokenization & Data Pipeline (`tokenizer/`, `data/`)**:
   - Pure Python Byte-Pair Encoding (BPE) tokenizer with byte-level fallback.
   - Zero-copy memory-mapped PyTorch Dataset (`np.memmap`) for massive text datasets.
   - Dynamic context chunking and lower-triangular causal attention masking.

2. **Core Decoder-Only Transformer (`models/`)**:
   - **Rotary Position Embeddings (RoPE)** with 2D complex frequency transformations (`rope.py`).
   - **Grouped-Query Attention (GQA)** supporting arbitrary Query-to-KV head ratios and integrated FlashAttention execution via `torch.nn.functional.scaled_dot_product_attention` (`attention.py`).
   - **KV Caching Container** (`KVCache`) for fast autoregressive generation (`attention.py`).
   - **Pre-normalization RMSNorm** and **SwiGLU Feed-Forward Network** (`layers.py`).
   - Complete `TransformerLM` architecture written directly in PyTorch (`transformer.py`).

3. **Distributed Engine & Custom Losses (`engine/`)**:
   - Autoregressive **Causal Cross-Entropy Loss** (`compute_causal_ce_loss`).
   - **Direct Preference Optimization (DPO)** Loss (`compute_dpo_loss`).
   - **Elastic Weight Consolidation (EWC)** Fisher Information Matrix estimation and penalty (`compute_ewc_penalty`).
   - **DistributedTrainer** supporting Automatic Mixed Precision (AMP `bfloat16`/`float16`), gradient accumulation, clipping, cosine annealing learning rate schedule with linear warmup, and PyTorch DDP scaffolding (`trainer.py`).

4. **Alignment & Continual Self-Learning Engine (`self_learning/`)**:
   - **Self-Instruct Generator**: Generates candidate task solutions using nucleus top-p sampling over seed prompts (`generator.py`).
   - **Execution Sandbox & Critique Evaluator**: Executes candidate Python code snippets in isolated buffers and scores responses to form $(x, y_w, y_l)$ preference pairs (`evaluator.py`).
   - **Baseline Replay Buffer**: Stores baseline pre-training tokens and updates diagonal Fisher Information matrix for catastrophic forgetting prevention (`replay_buffer.py`).

5. **Autonomous Orchestrator (`main.py`)**:
   - End-to-end execution loop linking all 5 subsystems into an autonomous self-improving agent.

---

## Mathematical Formulations

### 1. Autoregressive Causal Cross-Entropy Loss
$$\mathcal{L}_{\text{CE}}(\theta) = -\frac{1}{T} \sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t}) = -\frac{1}{T} \sum_{t=1}^{T} \log \left( \frac{\exp(z_{t, x_t})}{\sum_{v=1}^{V} \exp(z_{t, v})} \right)$$

### 2. Direct Preference Optimization (DPO) Loss
$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

### 3. Elastic Weight Consolidation (EWC) Loss
$$\mathcal{L}_{\text{total}}(\theta) = \mathcal{L}_{\text{DPO}}(\theta) + \sum_{i} \frac{\lambda}{2} F_i (\theta_i - \theta_{A, i}^*)^2$$

where $F_i$ is the empirical diagonal Fisher Information element computed over baseline replay data:
$$F_i = \mathbb{E}_{x \sim \mathcal{D}_{\text{base}}} \left[ \left( \frac{\partial \mathcal{L}_{\text{CE}}(x; \theta_A^*)}{\partial \theta_i} \right)^2 \right]$$

---

## Execution Instructions

To execute the end-to-end self-improvement loop:

```bash
cd "d:\Github projects\self_learning_llm"
python main.py
```

To run distributed multi-GPU training with PyTorch DDP:

```bash
NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 main.py
```
