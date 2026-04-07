import os
import json
import re
import subprocess
import sys

# Ensure all relative paths resolve correctly regardless of where the script is invoked from
os.chdir(os.path.dirname(os.path.abspath(__file__)))

subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "-q"])

import matplotlib.pyplot as plt
from pydantic import BaseModel, Field
import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
load_dotenv()

# --- SPEED OPTIMIZATIONS ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --- MODEL PATH ---
BEST_MODEL = "best_multilayer_injector.pt"

# --- LOAD LLAMA (FROZEN) ---
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN")

tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="sdpa", token=HF_TOKEN
)
model.eval()
model.requires_grad_(False)
device = model.device

# =============================================================================
# INJECTOR ARCHITECTURE & SAFE LOAD
# =============================================================================

class SingleLayerInjector(nn.Module):
    def __init__(self, dim=4096, bottleneck_dim=256, dropout_p=0.05):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=True, dropout=dropout_p)
        self.proj = nn.Sequential(
            nn.Linear(dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(bottleneck_dim, dim)
        )
        self.final_dropout = nn.Dropout(dropout_p)
        self.alpha_gate = nn.Sequential(
            nn.Linear(dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.max_alpha = 5.0

    def forward(self, query, keys, padding_mask):
        q_norm = self.norm_q(query)
        k_norm = self.norm_k(keys)
        attn_out, _ = self.attn(q_norm, k_norm, k_norm, key_padding_mask=padding_mask, need_weights=False)
        bottleneck_out = self.final_dropout(self.proj(attn_out))
        dynamic_alpha = self.alpha_gate(query) * self.max_alpha
        return dynamic_alpha * bottleneck_out


class MultiLayerInjector(nn.Module):
    def __init__(self, dim=4096, bottleneck_dim=256, dropout_p=0.05):
        super().__init__()
        self.layer_8 = SingleLayerInjector(dim, bottleneck_dim, dropout_p)
        self.layer_16 = SingleLayerInjector(dim, bottleneck_dim, dropout_p)
        self.layer_24 = SingleLayerInjector(dim, bottleneck_dim, dropout_p)


injector = MultiLayerInjector(dim=4096, bottleneck_dim=256, dropout_p=0.0).to(device, dtype=torch.bfloat16)

print(f"Loading best weights from {BEST_MODEL}...")

raw_state_dict = torch.load(BEST_MODEL, map_location=device, weights_only=True)
clean_state_dict = {}
for key, value in raw_state_dict.items():
    clean_key = key.replace('_orig_mod.', '') if key.startswith('_orig_mod.') else key
    clean_state_dict[clean_key] = value

injector.load_state_dict(clean_state_dict)
injector.eval()

# --- LOAD DATASETS & VECTORS (FULL DATA) ---
print("Loading Full Datasets & Vectors...")
train_df = pd.read_csv("train_tasks_expanded.csv")
test_df = pd.read_csv("test_tasks_expanded.csv")

train_df = train_df.dropna(subset=['target_response']).reset_index(drop=True)
test_df = test_df.dropna(subset=['target_response']).reset_index(drop=True)

train_vecs = torch.load("train_vectors_unified.pt", weights_only=False)
test_vecs = torch.load("test_vectors_unified.pt", weights_only=False)

# =============================================================================
# GENERATION HOOKS (PREFILL ONLY)
# =============================================================================

ACTIVE_SPONGE_IDX = None
ACTIVE_MASK = None
ACTIVE_KV_8 = None
ACTIVE_KV_16 = None
ACTIVE_KV_24 = None


def create_generation_hook(layer_name):
    def hook(module, args, output):
        hs = output[0] if isinstance(output, tuple) else output
        seq_len = hs.shape[1]
        if seq_len > 1 and ACTIVE_SPONGE_IDX < seq_len:
            query = hs[:, ACTIVE_SPONGE_IDX, :].unsqueeze(1)

            if layer_name == "layer_8":   injection = injector.layer_8(query, ACTIVE_KV_8, ACTIVE_MASK)
            elif layer_name == "layer_16": injection = injector.layer_16(query, ACTIVE_KV_16, ACTIVE_MASK)
            elif layer_name == "layer_24": injection = injector.layer_24(query, ACTIVE_KV_24, ACTIVE_MASK)

            hs_mod = hs.clone()
            hs_mod[:, ACTIVE_SPONGE_IDX, :] += injection.squeeze(1)
            return (hs_mod,) + output[1:] if isinstance(output, tuple) else hs_mod

        return output
    return hook


h8  = model.model.layers[8].register_forward_hook(create_generation_hook("layer_8"))
h16 = model.model.layers[16].register_forward_hook(create_generation_hook("layer_16"))
h24 = model.model.layers[24].register_forward_hook(create_generation_hook("layer_24"))

# =============================================================================
# INFERENCE ENGINE
# =============================================================================

def generate_steered_response(row, vecs_dict):
    global ACTIVE_KV_8, ACTIVE_KV_16, ACTIVE_KV_24, ACTIVE_MASK, ACTIVE_SPONGE_IDX

    c_id = row['question_id']
    prompt = row['inj_prompt']

    data = vecs_dict[c_id]
    chunks = data['chunk_signals'].to(device)
    gl = data['global_last'].to(device)

    kv_8_list, kv_16_list, kv_24_list = [], [], []
    for l_idx, lst in zip([0, 1, 2], [kv_8_list, kv_16_list, kv_24_list]):
        mu = chunks[:, l_idx, 0, :]
        max_vec = chunks[:, l_idx, 2, :]
        layer_gl = gl[l_idx, :].unsqueeze(0)

        kv_seq = torch.cat([mu, max_vec, layer_gl], dim=0)
        lst.append(kv_seq)

    seq_len = kv_8_list[0].shape[0]

    ACTIVE_KV_8  = kv_8_list[0].unsqueeze(0)
    ACTIVE_KV_16 = kv_16_list[0].unsqueeze(0)
    ACTIVE_KV_24 = kv_24_list[0].unsqueeze(0)
    ACTIVE_MASK  = torch.zeros((1, seq_len), dtype=torch.bool, device=device)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    sponge_id = tokenizer("...", add_special_tokens=False).input_ids[-1]

    sponge_matches = (inputs.input_ids[0] == sponge_id).nonzero(as_tuple=True)[0]
    ACTIVE_SPONGE_IDX = sponge_matches[-1].item() if len(sponge_matches) > 0 else inputs.input_ids.shape[1] - 1

    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        gens = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(gens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return generated_text

# =============================================================================
# BATCH EXECUTION & SAVING
# =============================================================================

def run_inference_and_save(df, vecs_dict, split_name, model_filename):
    print(f"\n{'='*80}")
    print(f"RUNNING INFERENCE: {split_name} SET | MODEL: {model_filename}")
    print(f"{'='*80}")

    generated_responses = []
    total = len(df)
    log_every = max(1, total // 10)

    output_csv_name = f"{split_name.lower()}_results_{model_filename.replace('.pt', '')}.csv"

    for i, row in df.iterrows():
        gen_resp = generate_steered_response(row, vecs_dict)
        generated_responses.append(gen_resp)
        if (i + 1) % log_every == 0 or (i + 1) == total:
            checkpoint_df = df.iloc[:len(generated_responses)].copy()
            checkpoint_df['generated_response'] = generated_responses
            checkpoint_df.to_csv(output_csv_name, index=False)
            print(f"  [{i + 1}/{total}] done — checkpoint saved to {output_csv_name}")

    print(f"Done. Final results in {output_csv_name}\n")


# --- FILTER TO QA TASK ONLY ---
train_df_qa = train_df[train_df['task_name'] == 'QA'].reset_index(drop=True)
test_df_qa  = test_df[test_df['task_name'] == 'QA'].reset_index(drop=True)
print(f"QA examples — Train: {len(train_df_qa)}, Test: {len(test_df_qa)}")

# --- LOOP OVER MODELS ---
models_to_test = ["best_multilayer_injector.pt", "final_multilayer_injector.pt"]

for model_path in models_to_test:
    print(f"\nLoading weights from {model_path}...")

    raw_state_dict = torch.load(model_path, map_location=device, weights_only=True)
    clean_state_dict = {}
    for key, value in raw_state_dict.items():
        clean_key = key.replace('_orig_mod.', '') if key.startswith('_orig_mod.') else key
        clean_state_dict[clean_key] = value

    injector.load_state_dict(clean_state_dict)
    injector.eval()

    run_inference_and_save(train_df_qa, train_vecs, "TRAIN", model_path)
    run_inference_and_save(test_df_qa,  test_vecs,  "TEST",  model_path)

print("All inference runs complete! 4 CSVs have been generated.")

# =============================================================================
# JUDGE MODEL INITIALIZATION
# =============================================================================

judge_model_id = "Qwen/Qwen2.5-3B-Instruct"

print(f"Loading Judge Model: {judge_model_id}...")
judge_tokenizer = AutoTokenizer.from_pretrained(judge_model_id, token=HF_TOKEN)
judge_tokenizer.pad_token = judge_tokenizer.eos_token

judge_model = AutoModelForCausalLM.from_pretrained(
    judge_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=HF_TOKEN
)

judge_model.eval()
judge_model.requires_grad_(False)


class EvaluationScores(BaseModel):
    similarity_check: float = Field(description="Score between 0.0 and 1.0")
    correctness_check: float = Field(description="Score between 0.0 and 1.0")
    context_adherence: float = Field(description="Score between 0.0 and 1.0")

# =============================================================================
# DYNAMIC PROMPTS & JUDGE LOGIC
# =============================================================================

def get_evaluation_prompt(task_name, instruction, context, gold_answer, generated_answer):
    base_prompt = f"""You are an expert evaluator for AI text generation systems.
Analyze the following data points and return a JSON object.

**Input Data:**
- Instruction/Prompt: "{instruction}"
- Context: "{context}"
- Gold Target: "{gold_answer}"
- Generated Output: "{generated_answer}"

**Task:**
Evaluate the 'Generated Output' on these 3 metrics using a float scale from 0.0 to 1.0 (where 0.0 is completely wrong/absent, and 1.0 is perfect):
"""

    if task_name == "QA":
        task_specific = """1. similarity_check: Score the semantic similarity between the Generated Output and the Gold Target.
2. correctness_check: Score how correctly the Generated Output answers the question posed in the Instruction.
3. context_adherence: Score how well the Generated Output relies ONLY on data/information derived from the provided Context."""

    elif task_name == "Summary":
        task_specific = """1. similarity_check: Score the semantic similarity between the Generated Output and the Gold Target summary.
2. correctness_check: Score how well the Generated Output captures the main points of the Context without omitting critical details.
3. context_adherence: Score how well the Generated Output avoids hallucinating external information not present in the Context."""

    elif task_name == "Repeat":
        task_specific = """1. similarity_check: Score the textual overlap and similarity between the Generated Output and the Gold Target.
2. correctness_check: Score how accurately the Generated Output repeats the requested information without modifying the meaning.
3. context_adherence: Score how strictly the Generated Output adheres to the exact phrasing or facts of the provided Context."""

    else:
        task_specific = """1. similarity_check: Score the semantic similarity between the Generated Output and the Gold Target.
2. correctness_check: Score how correctly the Generated Output fulfills the Instruction.
3. context_adherence: Score how well the Generated Output relies ONLY on the provided Context."""

    formatting = """
**Output Format:**
Return ONLY a valid JSON object with exact keys: "similarity_check", "correctness_check", "context_adherence"."""

    return base_prompt + task_specific + formatting


def local_qwen_judge(task_name, instruction, context, gold_answer, generated_answer):
    prompt = get_evaluation_prompt(task_name, instruction, context, gold_answer, generated_answer)

    messages = [
        {"role": "system", "content": "You are a strict JSON output machine. You only output valid JSON. No explanations."},
        {"role": "user", "content": prompt}
    ]

    text = judge_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = judge_tokenizer(text, return_tensors="pt").to(judge_model.device)

    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        outputs = judge_model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.0,
            do_sample=False,
            pad_token_id=judge_tokenizer.eos_token_id
        )

    response_text = judge_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    try:
        json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
        else:
            data = json.loads(response_text)

        s_check = max(0.0, min(1.0, float(data.get('similarity_check', 0.0))))
        c_check = max(0.0, min(1.0, float(data.get('correctness_check', 0.0))))
        a_check = max(0.0, min(1.0, float(data.get('context_adherence', 0.0))))

    except Exception as e:
        print(f"\nJSON Parsing error: {e} | Task: {task_name} | Raw Output: {response_text}")
        s_check, c_check, a_check = 0.0, 0.0, 0.0

    return s_check, c_check, a_check

# =============================================================================
# BATCH EVALUATION & SAVING CSVs
# =============================================================================

files_to_evaluate = [
    "train_results_best_multilayer_injector.csv",
    "test_results_best_multilayer_injector.csv",
    "train_results_final_multilayer_injector.csv",
    "test_results_final_multilayer_injector.csv"
]

evaluated_files = []


def clean_instruction(inj_prompt, task_name):
    if task_name == "QA":
        if "Question: " in inj_prompt:
            return inj_prompt.split("Question: ")[-1].split("\n")[0].strip()
        return "Answer the question based on the context."
    elif task_name == "Repeat":
        return "Repeat the context word by word like a paragraph."
    elif task_name == "Summary":
        return "Summarize the provided context in one short paragraph. Absolutely no bullet points or lists."
    return inj_prompt


for file_path in files_to_evaluate:
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}. Skipping.")
        continue

    print(f"\n{'='*80}")
    print(f"EVALUATING: {file_path}")
    print(f"{'='*80}")

    df = pd.read_csv(file_path)

    # --- FILTER TO QA TASK ONLY ---
    df = df[df['task_name'] == 'QA'].reset_index(drop=True)
    print(f"Evaluating {len(df)} QA examples from {file_path}")

    sim_scores, cor_scores, adh_scores = [], [], []
    total = len(df)
    log_every = max(1, total // 10)

    for i, row in df.iterrows():
        task = str(row['task_name'])
        dirty_prompt = str(row['inj_prompt'])
        clean_prompt = clean_instruction(dirty_prompt, task)

        s, c, a = local_qwen_judge(
            task_name=task,
            instruction=clean_prompt,
            context=str(row['context']),
            gold_answer=str(row['target_response']),
            generated_answer=str(row['generated_response'])
        )
        sim_scores.append(s)
        cor_scores.append(c)
        adh_scores.append(a)
        if (i + 1) % log_every == 0 or (i + 1) == total:
            checkpoint_df = df.iloc[:len(sim_scores)].copy()
            checkpoint_df['similarity_check'] = sim_scores
            checkpoint_df['correctness_check'] = cor_scores
            checkpoint_df['context_adherence'] = adh_scores
            out_name = file_path.replace(".csv", "_evaluated.csv")
            checkpoint_df.to_csv(out_name, index=False)
            print(f"  [{i + 1}/{total}] done — checkpoint saved to {out_name}")

    evaluated_files.append(out_name)
    print(f"Done. Final results in {out_name}")

# =============================================================================
# PLOTTING THE RESULTS
# =============================================================================

def plot_evaluation_scores(csv_path):
    df = pd.read_csv(csv_path)

    score_columns = ['similarity_check', 'correctness_check', 'context_adherence']
    averages = df[score_columns].mean()

    plt.figure(figsize=(8, 6))
    bars = plt.bar(
        ['Similarity', 'Correctness', 'Adherence'],
        averages.values,
        color=['#1f77b4', '#ff7f0e', '#2ca02c'],
        edgecolor='black'
    )

    plt.ylim(0, 1.1)

    title_text = os.path.basename(csv_path).replace("_evaluated.csv", "").replace("_", " ").title()
    plt.title(f'Average Judge Scores: {title_text}')
    plt.ylabel('Average Score (0-1)')
    plt.xlabel('Metrics')

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            yval + 0.02,
            f'{yval:.3f}',
            ha='center', va='bottom', fontweight='bold'
        )

    plt.tight_layout()
    plt.savefig(csv_path.replace(".csv", "_plot.png"), dpi=150)
    plt.show()


print("\nGenerating Plots...")
for eval_file in evaluated_files:
    plot_evaluation_scores(eval_file)
