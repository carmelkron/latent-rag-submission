# Latent-RAG: Frozen but Not Fixed

**Learning to Steer LLMs with External Knowledge via Activation Injection**

Carmel Kronfeld\* and Eran Tascesme\*  
School of Industrial & Intelligent Systems Engineering, Tel Aviv University  
\*Equal contribution

---

## 🔍 Overview

Latent-RAG injects external knowledge directly into a frozen LLM via learned activation steering, bypassing the context window entirely. This eliminates the structural vulnerability of standard RAG to indirect prompt injection, where retrieved text and system instructions share the same token stream.

A lightweight injector network maps unseen knowledge — extracted from either token embeddings or residual-stream hidden states — into a steering vector added to the model's activations at inference time, with **no weight modifications** to the base model. This creates a **structurally separate channel** for knowledge delivery — distinct from the instruction stream — which standard RAG cannot achieve.

The method works in three stages:

1. **Extract** an internal representation of the target knowledge — either from the model's token embeddings or from residual-stream hidden states at a chosen layer
2. **Train** a lightweight injector network to map these representations into effective steering vectors
3. **Inject** the learned steering vector into the residual stream at inference time, causing the model to generate responses as if it had the knowledge in context

<p align="center">
  <img src="figures/figure1.png" alt="Latent-RAG method overview" width="700"/>
</p>

### 📊 Key Results

| Experiment | Metric | Score |
|---|---|---|
| Entity injection (curated benchmark) | Entity recall | 42.2% |
| Entity injection (curated benchmark) | Factual accuracy | 40.3% |
| Entity injection (PopQA) | Accuracy | 32.4% (vs. 30.2% oracle RAG) |
| Sentence injection (unseen passages) | Semantic similarity | 76.8% |
| Sentence injection (unseen passages) | Factual correctness | 56.4% |

## 📦 What's Included (and What's Not)

| ✅ Included in this repo | ❌ Not included (generated at runtime) |
|---|---|
| All experiment notebooks and scripts | Model weights (`.pt` files, ~20GB total) |
| Training and evaluation code | Expanded CSV datasets (~160MB) |
| Pre-computed result CSVs and plots | Activation vector caches |
| Paper figures | Llama-3.1-8B-Instruct model (downloaded via HF) |
| PopQA benchmark data | |
| Requirements and setup files | |

All excluded artifacts are **regenerated automatically** when you run the notebooks in order.

## 🗂️ Repository Structure

```
latent-rag-submission/
├── figures/                          # Paper figures
│   ├── figure1.png
│   ├── concept_category_barplot.png
│   └── introspection_layer_sweep.png
│
├── anthropic-paper-replication/      # Replication of Anthropic's activation steering
│   ├── Paper_Replication.ipynb       # Concept extraction, injection, evaluation
│   └── results/                      # Per-layer injection results (CSVs) and plots
│
├── injecting-words/                  # Entity-level injection experiments
│   ├── Injecting_Words_V4.ipynb      # ⭐ Latest entity injection (SFT + GRPO)
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
│   ├── training.py                   # Training script (cluster job submission)
│   ├── Inference.ipynb               # Run inference with trained injector
│   ├── Inference_LLM_as_Judge.ipynb  # LLM-as-judge evaluation
│   ├── inference_llm_as_judge.py     # Evaluation script (cluster job submission)
│   └── old_experiment_one_injector/  # Earlier single-injector experiments
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

> **Note on `.py` files:** The Python scripts (`training.py`, `inference_llm_as_judge.py`) contain the same logic as their corresponding notebooks (`Training.ipynb`, `Inference_LLM_as_Judge.ipynb`). They were used to submit non-interactive jobs to a GPU cluster. If you're exploring the code, the notebooks are the easier entry point.

## 🧬 Architecture Versions (Entity Injection)

The entity injection experiments evolved through several architecture iterations. Each version is preserved in its own notebook for reproducibility:

| Version | Architecture | Training | Key Change |
|---|---|---|---|
| Previous | 2-layer MLP / 4-layer bottleneck MLP | SFT (CE loss) | Baseline — first injection experiments |
| V1 | Same as Previous | SFT (CE loss) | Bug fixes in injection vector indexing |
| V2 | Residual bottleneck adapter (512 dim) | SFT (weighted CE) → GRPO | Entity token upweighting (5x), added GRPO phase |
| V3 | Arch 3 (bottleneck) vs Arch 4 (2-layer MLP, 4096 dim) | SFT only (weighted CE + BERTTune) | Added semantic loss, entity weight 8x, dual-arch comparison |
| **V4** ⭐ | **2-layer MLP (4096 dim)** | **SFT → GRPO with Qwen judge** | **Best arch from V3 + Qwen-2.5-3B judge + KL penalty** |
| TokenEmbed | Same as V4 | Same as V4 | Concept vectors from token embeddings instead of residual states |

## 🧬 Architecture Versions (Sentence Injection)

| Version | Architecture | Injection | Training | Key Change |
|---|---|---|---|---|
| Old (`old_experiment_one_injector/`) | `TriContextInjector` — self-attention mixer + linear projection | Single layer (layer 16), replacement | SFT → RL (two-phase) | Baseline — learned position weights over context vectors |
| **Current** ⭐ | **`MultiLayerInjector` — cross-attention (8 heads) + bottleneck MLP** | **3 layers (8, 16, 24), additive** | **SFT only (100 epochs)** | **Multi-layer injection, dynamic alpha gating, no RL needed** |

## ⚙️ Setup

### Requirements

- Python 3.10+
- CUDA-capable GPU (experiments were run on NVIDIA A100 and H100)
- ~40GB+ GPU memory recommended (Llama-3.1-8B + Qwen-2.5-3B judge + injector + data on device during GRPO training)

```bash
pip install -r requirements.txt
```

### 🔑 API Keys

After cloning the repo, create a `.env` file in the root directory:

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

All experiments use [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct). You need to request access to this gated model on Hugging Face before running any notebooks.

## 🚀 Reproducing Results

### 1. Anthropic Paper Replication (Appendix A)

Replication of the activation steering approach from Anthropic's work, establishing that concept representations exist in the residual stream and can be steered.

```
cd anthropic-paper-replication
jupyter notebook Paper_Replication.ipynb
```

### 2. Entity-Level Injection

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

### 3. Sentence-Level Injection

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

> ⚠️ **Steps 2–3 generate large intermediate files** that are excluded from this repo:
> - `Dataset_Creation.ipynb` produces `train_tasks_expanded.csv` (~125MB) and `test_tasks_expanded.csv` (~32MB)
> - `Extraction.ipynb` produces `train_vectors_unified.pt` (~15GB) and `test_vectors_unified.pt` (~3.7GB)
>
> These must be generated before running `Training.ipynb`.
