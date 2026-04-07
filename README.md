# Latent-RAG: Frozen but Not Fixed

**Learning to Steer LLMs with External Knowledge via Activation Injection**

Carmel Kronfeld\* and Eran Tascesme\*  
School of Industrial & Intelligent Systems Engineering, Tel Aviv University  
\*Equal contribution

---

## Overview

Latent-RAG injects external knowledge directly into a frozen LLM's residual stream via learned activation steering, bypassing the context window entirely. This eliminates the structural vulnerability of standard RAG to indirect prompt injection, where retrieved text and system instructions share the same token stream.

A lightweight injector network maps unseen knowledge into a steering vector added to the model's hidden states at inference time, with **no weight modifications** to the base model.

### Key Results

| Experiment | Metric | Score |
|---|---|---|
| Entity injection (curated benchmark) | Entity recall | 42.2% |
| Entity injection (curated benchmark) | Factual accuracy | 40.3% |
| Entity injection (PopQA) | Accuracy | 32.4% (vs. 30.2% oracle RAG) |
| Sentence injection (unseen passages) | Semantic similarity | 76.8% |
| Sentence injection (unseen passages) | Factual correctness | 56.4% |

## Repository Structure

```
latent-rag-submission/
├── figures/                          # Paper figures
│   ├── concept_category_barplot.png
│   ├── figure1.png
│   └── introspection_layer_sweep.png
│
├── anthropic-paper-replication/      # Replication of Anthropic's activation steering
│   ├── Paper_Replication.ipynb       # Main notebook: concept extraction, injection, evaluation
│   └── results/                      # Per-layer injection results (CSVs) and plots
│
├── injecting-words/                  # Entity-level injection experiments
│   ├── Injecting_Words_V4.ipynb      # Latest entity injection (architecture 4, SFT + GRPO)
│   ├── Injecting_Words_TokenEmbed_V1.ipynb  # Token embedding baseline
│   ├── RAG_Baseline_V4.ipynb         # RAG baseline comparison
│   ├── PopQA_Benchmark.ipynb         # PopQA evaluation (V1)
│   ├── PopQA_V2.ipynb                # PopQA evaluation (V2, final)
│   ├── Injecting_Words_V3.ipynb      # Earlier architecture iteration
│   ├── Injecting_Words_V2.ipynb      # Earlier architecture iteration
│   ├── Injecting_Words_V1.ipynb      # Earlier architecture iteration
│   ├── Injecting_Words_Previous.ipynb # Initial architecture exploration
│   ├── results_v4/                   # Results for V4 experiments
│   ├── results_v3/                   # Results for V3 experiments
│   ├── results_v1/                   # Results for V1/V2 experiments
│   ├── results_previous/             # Results for initial experiments
│   ├── PopQA/                        # PopQA V1 dataset, results, and plots
│   └── PopQA_V2/                     # PopQA V2 dataset, results, and plots
│
├── injecting-sentences/              # Sentence-level injection experiments
│   ├── Dataset_Creation.ipynb        # Create training/test datasets
│   ├── RepliQA_Dataset.ipynb         # RepliQA dataset preparation
│   ├── Extraction.ipynb              # Extract activation vectors from passages
│   ├── Training.ipynb                # Train multi-layer cross-attention injector
│   ├── training.py                   # Standalone training script
│   ├── Inference.ipynb               # Run inference with trained injector
│   ├── Inference_LLM_as_Judge.ipynb  # LLM-as-judge evaluation
│   ├── inference_llm_as_judge.py     # Standalone evaluation script
│   └── old_experiment_one_injector/  # Earlier single-injector experiments
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

### Requirements

- Python 3.10+
- CUDA-capable GPU (experiments were run on NVIDIA A100)
- ~16GB GPU memory for Llama-3.1-8B-Instruct

```bash
pip install -r requirements.txt
```

### API Keys

After cloning the repo, create a `.env` file in the root directory with your API keys:

```bash
cp .env.example .env
```

Then edit `.env` and fill in your keys:

```
HF_TOKEN=hf_your_token_here
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_API_KEY=your_gemini_api_key_here
```

| Key | Required for | How to get |
|-----|-------------|------------|
| `HF_TOKEN` | Downloading Llama-3.1-8B-Instruct | [HuggingFace tokens](https://huggingface.co/settings/tokens) |
| `GEMINI_API_KEY` | LLM-as-judge evaluation cells | [Google AI Studio](https://aistudio.google.com/apikey) |
| `GOOGLE_API_KEY` | Same as above (used in older notebooks) | Same as above |

### Model

All experiments use [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct). You will need access to this model on Hugging Face.

## Reproducing Results

### 1. Anthropic Paper Replication (Section 3)

Replication of the activation steering approach from Anthropic's work, establishing that concept representations exist in the residual stream and can be steered.

```
cd anthropic-paper-replication
jupyter notebook Paper_Replication.ipynb
```

### 2. Entity-Level Injection (Section 4.1)

The core Latent-RAG experiment: training an injector MLP to encode entity knowledge as residual-stream perturbations.

**Recommended starting point** (final version):
```
cd injecting-words
jupyter notebook Injecting_Words_V4.ipynb     # Main experiment
jupyter notebook PopQA_V2.ipynb                # PopQA benchmark evaluation
jupyter notebook RAG_Baseline_V4.ipynb         # RAG baseline for comparison
```

The notebook pipeline:
1. Extract concept vectors for entities at layer 16
2. Train a two-layer MLP adapter (Phase 1: SFT, Phase 2: GRPO reinforcement learning)
3. Run inference: inject steering vectors and evaluate model responses
4. LLM-as-judge evaluation of generated answers

### 3. Sentence-Level Injection (Section 4.2)

Injecting full factual passages via a cross-attention-based multi-layer injector.

```
cd injecting-sentences
jupyter notebook RepliQA_Dataset.ipynb          # Step 1: Prepare RepliQA source data
jupyter notebook Dataset_Creation.ipynb         # Step 2: Create expanded datasets
jupyter notebook Extraction.ipynb               # Step 3: Extract activation vectors
jupyter notebook Training.ipynb                 # Step 4: Train injector
jupyter notebook Inference.ipynb                # Step 5: Run inference
jupyter notebook Inference_LLM_as_Judge.ipynb   # Step 6: Evaluate with LLM judge
```

**Important:** Steps 2-3 generate large intermediate files that are excluded from this repo:
- `Dataset_Creation.ipynb` produces `train_tasks_expanded.csv` (~125MB) and `test_tasks_expanded.csv` (~32MB) — expanded task datasets with generated prompts
- `Extraction.ipynb` produces `train_vectors_unified.pt` (~15GB) and `test_vectors_unified.pt` (~3.7GB) — extracted activation vectors

These must be generated before running Training.ipynb.

### Binary Artifacts

Model weights (`.pt` files) and cached activation vectors are excluded from this repository due to their size (~20GB total). They will be regenerated when you run the notebooks. The key artifacts:

- **Activation caches**: Extracted residual-stream vectors for entities/passages
- **Trained adapters**: Injector MLP or cross-attention network weights
- **Unified vectors**: Pre-extracted vectors for the full dataset

## Method

Latent-RAG works by:

1. **Extracting** a concept's residual-stream representation by running a prompt through the frozen LLM and reading the hidden state at a target layer
2. **Training** a lightweight injector network to map these representations into effective steering vectors
3. **Injecting** the learned steering vector into the residual stream at inference time, causing the model to generate responses as if it had the knowledge in context

This creates a **structurally separate channel** for knowledge, distinct from the instruction stream, which standard RAG cannot achieve.

## Citation

```bibtex
@article{kronfeld2025latentrag,
  title={Frozen but Not Fixed: Learning to Steer LLMs with External Knowledge},
  author={Kronfeld, Carmel and Tascesme, Eran},
  year={2025}
}
```

## License

This repository is released for academic and research purposes.
