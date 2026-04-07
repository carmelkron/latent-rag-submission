import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
load_dotenv()

# --- SPEED OPTIMIZATIONS ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.cufft_plan_cache.clear()

# --- SETUP & PATHS ---
BEST_MODEL = "best_multilayer_injector.pt"
FINAL_MODEL = "final_multilayer_injector.pt"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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

# --- LOAD UNIFIED DATASETS & VECTORS ---
print("Loading Unified Datasets & Tensors...")
train_df = pd.read_csv(os.path.join(BASE_DIR, "train_tasks_expanded.csv")).dropna(subset=['question_id', 'task_name', 'inj_prompt', 'target_response'])
test_df = pd.read_csv(os.path.join(BASE_DIR, "test_tasks_expanded.csv")).dropna(subset=['question_id', 'task_name', 'inj_prompt', 'target_response'])

train_vecs = torch.load(os.path.join(BASE_DIR, "train_vectors_unified.pt"), weights_only=False)
test_vecs = torch.load(os.path.join(BASE_DIR, "test_vectors_unified.pt"), weights_only=False)

# --- MULTI-LAYER ARCHITECTURE ---
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

# --- RESUME LOGIC ---
if os.path.exists(BEST_MODEL):
    print(f"\n🚀 FOUND PREVIOUS SAVE: Loading weights from {BEST_MODEL}...")
    raw_state_dict = torch.load(BEST_MODEL, map_location=device, weights_only=True)
    clean_state_dict = {}
    for key, value in raw_state_dict.items():
        clean_key = key.replace('_orig_mod.', '') if key.startswith('_orig_mod.') else key
        clean_state_dict[clean_key] = value
    injector.load_state_dict(clean_state_dict)
    print("✅ Weights loaded successfully.")

print("Compiling injector architecture for maximum A100 speed...")
injector = torch.compile(injector)

optimizer = torch.optim.AdamW(injector.parameters(), lr=5e-4, weight_decay=0.0001, fused=True)

# --- GLOBAL VARIABLES & FORWARD HOOKS ---
ACTIVE_SPONGE_IDX = None
ACTIVE_MASK = None
ACTIVE_KV_8 = None
ACTIVE_KV_16 = None
ACTIVE_KV_24 = None

def create_hook(layer_name):
    def hook(module, args, output):
        hs = output[0] if isinstance(output, tuple) else output
        B = hs.shape[0]
        batch_indices = torch.arange(B, device=hs.device)
        query = hs[batch_indices, ACTIVE_SPONGE_IDX, :].unsqueeze(1)

        if layer_name == "layer_8": injection = injector.layer_8(query, ACTIVE_KV_8, ACTIVE_MASK)
        elif layer_name == "layer_16": injection = injector.layer_16(query, ACTIVE_KV_16, ACTIVE_MASK)
        elif layer_name == "layer_24": injection = injector.layer_24(query, ACTIVE_KV_24, ACTIVE_MASK)

        hs_mod = hs.clone()
        hs_mod[batch_indices, ACTIVE_SPONGE_IDX, :] += injection.squeeze(1)
        return (hs_mod,) + output[1:] if isinstance(output, tuple) else hs_mod
    return hook

h8 = model.model.layers[8].register_forward_hook(create_hook("layer_8"))
h16 = model.model.layers[16].register_forward_hook(create_hook("layer_16"))
h24 = model.model.layers[24].register_forward_hook(create_hook("layer_24"))

# --- DATASET & DATALOADER ---
class SteeringDataset(Dataset):
    def __init__(self, df, vecs_dict):
        self.df = df.reset_index(drop=True)
        self.vecs = vecs_dict
        self.sponge_id = tokenizer("...", add_special_tokens=False).input_ids[-1]

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        full_text = str(row['inj_prompt']) + " " + str(row['target_response']) + tokenizer.eos_token
        inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs.input_ids[0]
        prompt_len = len(tokenizer(str(row['inj_prompt']), add_special_tokens=False).input_ids)
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        sponge_matches = (input_ids == self.sponge_id).nonzero(as_tuple=True)[0]
        sponge_idx = sponge_matches[0].item() if len(sponge_matches) > 0 else prompt_len - 1
        return input_ids, labels, sponge_idx, row['question_id'], row['task_name']

def custom_collate(batch):
    input_ids = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    sponge_indices = [item[2] for item in batch]
    c_ids = [item[3] for item in batch]
    task_names = [item[4] for item in batch]

    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    return input_ids_padded, labels_padded, sponge_indices, c_ids, task_names

train_loader = DataLoader(
    SteeringDataset(train_df, train_vecs),
    batch_size=14,
    shuffle=True,
    collate_fn=custom_collate,
    num_workers=16,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True
)

val_loader = DataLoader(
    SteeringDataset(test_df, test_vecs),
    batch_size=14,
    shuffle=False,
    collate_fn=custom_collate,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)

# --- BATCH PREPARATION ---
def prepare_batch_vectors(c_ids, vecs_dict, is_training):
    global ACTIVE_KV_8, ACTIVE_KV_16, ACTIVE_KV_24, ACTIVE_MASK
    B = len(c_ids)
    kv_8_list, kv_16_list, kv_24_list, lengths = [], [], [], []

    for cid in c_ids:
        data = vecs_dict[cid]
        chunks = data['chunk_signals'].to(device, non_blocking=True)
        gl = data['global_last'].to(device, non_blocking=True)

        for l_idx, lst in zip([0, 1, 2], [kv_8_list, kv_16_list, kv_24_list]):
            mu, var, max_vec = chunks[:, l_idx, 0, :], chunks[:, l_idx, 1, :], chunks[:, l_idx, 2, :]
            if is_training:
                std = torch.sqrt(var.clamp(min=1e-6))
                z = mu + torch.randn_like(mu) * std
            else:
                z = mu
            layer_gl = gl[l_idx, :].unsqueeze(0)
            kv_seq = torch.cat([z, max_vec, layer_gl], dim=0)
            lst.append(kv_seq)
        lengths.append(kv_8_list[-1].shape[0])

    max_len = max(lengths)
    ACTIVE_KV_8 = torch.zeros((B, max_len, 4096), dtype=torch.bfloat16, device=device)
    ACTIVE_KV_16 = torch.zeros((B, max_len, 4096), dtype=torch.bfloat16, device=device)
    ACTIVE_KV_24 = torch.zeros((B, max_len, 4096), dtype=torch.bfloat16, device=device)
    ACTIVE_MASK = torch.ones((B, max_len), dtype=torch.bool, device=device)

    for i in range(B):
        l = lengths[i]
        ACTIVE_KV_8[i, :l, :] = kv_8_list[i]
        ACTIVE_KV_16[i, :l, :] = kv_16_list[i]
        ACTIVE_KV_24[i, :l, :] = kv_24_list[i]
        ACTIVE_MASK[i, :l] = False

# --- TRAINING LOOP ---
epochs = 100
best_val_loss = float('inf')
patience, max_patience = 0, 50

loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
loss_fn_none = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

for epoch in range(epochs):
    injector.train()
    train_loss = 0
    t_task_loss = {'QA': 0.0, 'Summary': 0.0, 'Repeat': 0.0}
    t_task_count = {'QA': 0, 'Summary': 0, 'Repeat': 0}

    total_train_batches = len(train_loader)
    for batch_idx, (input_ids, labels, sponge_indices, c_ids, task_names) in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)

        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        ACTIVE_SPONGE_IDX = torch.tensor(sponge_indices, device=device)

        prepare_batch_vectors(c_ids, train_vecs, is_training=True)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model(input_ids)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            with torch.no_grad():
                token_losses = loss_fn_none(shift_logits.transpose(1, 2), shift_labels)
                for i, task in enumerate(task_names):
                    valid_tokens = (shift_labels[i] != -100).sum()
                    if valid_tokens > 0:
                        seq_loss = token_losses[i].sum() / valid_tokens
                        if task in t_task_loss:
                            t_task_loss[task] += seq_loss.item()
                            t_task_count[task] += 1

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if (batch_idx + 1) % 1000 == 0 or (batch_idx + 1) == total_train_batches:
            avg_so_far = train_loss / (batch_idx + 1)
            print(f"  [Epoch {epoch+1} | Batch {batch_idx+1}/{total_train_batches}] Loss: {avg_so_far:.4f}")

    # --- VALIDATION ---
    injector.eval()
    val_loss = 0
    v_task_loss = {'QA': 0.0, 'Summary': 0.0, 'Repeat': 0.0}
    v_task_count = {'QA': 0, 'Summary': 0, 'Repeat': 0}

    total_val_batches = len(val_loader)
    with torch.no_grad():
        for batch_idx, (input_ids, labels, sponge_indices, c_ids, task_names) in enumerate(val_loader):
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            ACTIVE_SPONGE_IDX = torch.tensor(sponge_indices, device=device)

            prepare_batch_vectors(c_ids, test_vecs, is_training=False)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(input_ids)
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                val_loss += loss.item()

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == total_val_batches:
                print(f"  [Validation | Batch {batch_idx+1}/{total_val_batches}]")

            token_losses = loss_fn_none(shift_logits.transpose(1, 2), shift_labels)
            for i, task in enumerate(task_names):
                valid_tokens = (shift_labels[i] != -100).sum()
                if valid_tokens > 0:
                    seq_loss = token_losses[i].sum() / valid_tokens
                    if task in v_task_loss:
                        v_task_loss[task] += seq_loss.item()
                        v_task_count[task] += 1

    avg_train = train_loss / len(train_loader)
    avg_val = val_loss / len(val_loader)

    print(f"\nEpoch {epoch+1}: Train Loss = {avg_train:.4f} | Val Loss = {avg_val:.4f}")

    t_qa = t_task_loss['QA'] / max(1, t_task_count['QA'])
    t_sum = t_task_loss['Summary'] / max(1, t_task_count['Summary'])
    t_rep = t_task_loss['Repeat'] / max(1, t_task_count['Repeat'])
    print(f"  --> Train Details | QA: {t_qa:.4f} | Summary: {t_sum:.4f} | Repeat: {t_rep:.4f}")

    v_qa = v_task_loss['QA'] / max(1, v_task_count['QA'])
    v_sum = v_task_loss['Summary'] / max(1, v_task_count['Summary'])
    v_rep = v_task_loss['Repeat'] / max(1, v_task_count['Repeat'])
    print(f"  --> Val Details   | QA: {v_qa:.4f} | Summary: {v_sum:.4f} | Repeat: {v_rep:.4f}")

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        patience = 0
        torch.save(injector.state_dict(), BEST_MODEL)
        print(f"🌟 Best model saved to Drive! (Val Loss: {best_val_loss:.4f})")
    else:
        patience += 1
        print(f"⚠️ No improvement. Patience: {patience}/{max_patience}")
        if patience >= max_patience:
            print("🛑 Early stopping triggered!")
            break

# --- SAVE FINAL MODEL ---
torch.save(injector.state_dict(), FINAL_MODEL)

# h8.remove()
# h16.remove()
# h24.remove()
