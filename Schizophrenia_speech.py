
# ===============================================================
# SCHIZOPHRENIA SEVERITY MODEL
# ===============================================================

# ===============================
# 1║ Mount Google Drive
# ===============================
from google.colab import drive
drive.mount('/content/drive')

# ===============================
# 2║ Install Dependencies
# ===============================
import subprocess
subprocess.run(["pip", "install", "faster-whisper", "transformers", "datasets",
                "scikit-learn", "openpyxl", "--quiet"], check=True)

# ===============================
# 3║ Imports
# ===============================
import os, zipfile, time, warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from faster_whisper import WhisperModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ===============================
# 4║ Config
# ===============================
BASE_PATH     = "/content/drive/MyDrive"
ZIP_PATH      = os.path.join(BASE_PATH, "audio_97.zip")
EXTRACT_DIR   = "/content/audio_extracted"
METADATA_PATH = os.path.join(BASE_PATH, "train_split_Depression.csv")
CACHE_PATH    = os.path.join(BASE_PATH, "transcription_cache.csv")  # saved to Drive

MAX_LENGTH      = 512
BATCH_SIZE      = 8
EPOCHS          = 6
N_FOLDS         = 5
LEARNING_RATE   = 2e-5
UNFREEZE_LAYERS = list(range(4, 12))
SEVERITY_ALPHA  = 0.4

# ===============================
# 5║ Extract ZIP
# ===============================
if not os.path.exists(EXTRACT_DIR):
    print("Extracting ZIP...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        zf.extractall(EXTRACT_DIR)
    print("Done.")
else:
    print("Already extracted.")

# Build audio map: pid -> path
SUPPORTED = {".wav", ".mp3", ".flac", ".m4a"}
audio_map = {}
for f in Path(EXTRACT_DIR).rglob("*"):
    if f.suffix.lower() in SUPPORTED:
        pid = f.stem.split("_")[0].strip()
        audio_map[pid] = str(f)
print(f"Audio files found: {len(audio_map)}")

# ===============================
# 6║ Metadata
# ===============================
metadata = pd.read_csv(METADATA_PATH)
metadata.columns = [c.strip().lower().replace(" ", "_") for c in metadata.columns]
metadata["participant_id"] = metadata["participant_id"].astype(str).str.strip()
metadata["label"]    = metadata["phq8_binary"].astype(int)
metadata["severity"] = metadata["phq8_score"].clip(0, 24) / 24.0
print(f"Participants: {len(metadata)} | Positive: {metadata['label'].sum()} | Negative: {(metadata['label']==0).sum()}")

# ===============================
# 7║ TRANSCRIPTION (GPU, Cached)
# ===============================

# Load cache — accepts BOTH "pid" and "participant_id" column names
# so it works whether cache was created by old or new code
cache = {}
if os.path.exists(CACHE_PATH):
    try:
        cache_df  = pd.read_csv(CACHE_PATH)
        id_col_c  = next((c for c in cache_df.columns if c in ("pid", "participant_id")), None)
        txt_col_c = next((c for c in cache_df.columns if c == "text"), None)
        if id_col_c and txt_col_c:
            cache = dict(zip(cache_df[id_col_c].astype(str), cache_df[txt_col_c].astype(str)))
            print(f"✓ Loaded {len(cache)} cached transcriptions from Drive.")
        else:
            print(f"Cache columns {cache_df.columns.tolist()} unrecognised — starting fresh.")
    except Exception as ex:
        print(f"Could not read cache ({ex}) — starting fresh.")
else:
    print("No cache found — will transcribe all files.")

def save_cache():
    """Always saves with 'pid' column so future runs load correctly."""
    pd.DataFrame({
        "pid":  list(cache.keys()),
        "text": list(cache.values())
    }).to_csv(CACHE_PATH, index=False)

# Only transcribe files not already in cache
need_transcription = [
    (str(row["participant_id"]), audio_map[str(row["participant_id"])])
    for _, row in metadata.iterrows()
    if str(row["participant_id"]) in audio_map
    and str(row["participant_id"]) not in cache
]
print(f"Need transcription: {len(need_transcription)} | Already cached: {len(cache)}")

if need_transcription:
    compute = "float16" if device.type == "cuda" else "int8"
    hw      = "cuda"   if device.type == "cuda" else "cpu"
    print(f"\nLoading faster-whisper 'medium' ({compute} on {hw})...")
    whisper_model = WhisperModel("medium", device=hw, compute_type=compute)
    print(f"Transcribing {len(need_transcription)} files...\n")

    t_start = time.time()
    for i, (pid, audio_path) in enumerate(need_transcription):
        t0 = time.time()
        try:
            segments, _ = whisper_model.transcribe(
                audio_path,
                language="en",
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )
            text = " ".join(s.text for s in segments).strip() or "[inaudible]"
        except Exception as ex:
            print(f"  ⚠ Error on {pid}: {ex}")
            text = "[error]"

        cache[pid] = text
        elapsed   = time.time() - t0
        remaining = (len(need_transcription) - i - 1) * elapsed
        print(f"  [{i+1}/{len(need_transcription)}] PID {pid} | {elapsed:.1f}s | "
              f"ETA: {remaining/60:.1f} min | {text[:70]}...")

        # Save to Drive every 10 files — progress safe even if session crashes
        if (i + 1) % 10 == 0:
            save_cache()
            print(f"  💾 Cache saved ({len(cache)} entries)")

    save_cache()
    total_time = time.time() - t_start
    print(f"\n✓ Transcription done in {total_time/60:.1f} min")
    del whisper_model
    torch.cuda.empty_cache()
    print("GPU memory freed for training.")
else:
    print("✓ All cached — skipping transcription.")

# ===============================
# 8║ Assemble Dataset
# ===============================
texts, labels, severities = [], [], []
skipped = []

for _, row in metadata.iterrows():
    pid  = str(row["participant_id"])
    text = cache.get(pid, "")
    if not text or text in ("[error]", "[inaudible]", "nan", ""):
        skipped.append(pid)
        continue
    texts.append(text)
    labels.append(int(row["label"]))
    severities.append(float(row["severity"]))

print(f"\n✓ Usable samples: {len(texts)}")
print(f"  Positive: {sum(labels)} | Negative: {len(labels)-sum(labels)}")
if skipped:
    print(f"  Skipped {len(skipped)} PIDs (no audio or bad transcript): {skipped}")

if len(texts) == 0:
    raise RuntimeError("No transcripts. Check cache or audio files.")

# ===============================
# 9║ Tokenizer
# ===============================
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_fn(batch):
    return tokenizer(batch["text"], padding="max_length",
                     truncation=True, max_length=MAX_LENGTH)

# ===============================
# 10║ Model
# ===============================
class AttentionPool(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states, mask):
        scores  = self.attn(hidden_states).squeeze(-1)
        scores  = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (hidden_states * weights).sum(dim=1)


class DualLinguisticModel(nn.Module):
    def __init__(self, bert, electra):
        super().__init__()
        self.bert         = bert
        self.electra      = electra
        dim               = 768
        self.bert_pool    = AttentionPool(dim)
        self.electra_pool = AttentionPool(dim)
        self.shared = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, 512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256),     nn.GELU(), nn.Dropout(0.2),
        )
        self.classifier    = nn.Linear(256, 2)
        self.severity_head = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask):
        b = self.bert(input_ids=input_ids,    attention_mask=attention_mask)
        e = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        shared = self.shared(torch.cat([
            self.bert_pool(b.last_hidden_state, attention_mask),
            self.electra_pool(e.last_hidden_state, attention_mask),
        ], dim=1))
        return self.classifier(shared), self.severity_head(shared)


def load_fresh_model():
    bert    = AutoModel.from_pretrained("bert-base-uncased").to(device)
    electra = AutoModel.from_pretrained("google/electra-base-discriminator").to(device)
    for enc in [bert, electra]:
        for p in enc.parameters(): p.requires_grad = False
        for i in UNFREEZE_LAYERS:
            for p in enc.encoder.layer[i].parameters(): p.requires_grad = True
    return DualLinguisticModel(bert, electra).to(device)

# ===============================
# 11║ Dataset Helpers
# ===============================
def make_dataset(t, l, s):
    ds = Dataset.from_dict({"text": t, "label": l, "severity": s})
    ds = ds.map(tokenize_fn, batched=True, batch_size=32)
    ds.set_format("torch", columns=["input_ids", "attention_mask", "label", "severity"])
    return ds

def collate_fn(batch):
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"]  for b in batch]),
        "label":          torch.tensor([b["label"]          for b in batch], dtype=torch.long),
        "severity":       torch.tensor([b["severity"]       for b in batch], dtype=torch.float),
    }

# ===============================
# 12║ Training
# ===============================
texts_np      = np.array(texts)
labels_np     = np.array(labels)
severities_np = np.array(severities)

skf      = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
loss_cls = nn.CrossEntropyLoss(label_smoothing=0.05)
loss_reg = nn.HuberLoss(delta=0.5)

all_accs, all_f1s = [], []
best_val_acc      = 0.0
best_model_state  = None

for fold, (train_idx, val_idx) in enumerate(skf.split(texts_np, labels_np)):
    print(f"\n{'='*55}")
    print(f"  FOLD {fold+1}/{N_FOLDS}  |  Train: {len(train_idx)}  Val: {len(val_idx)}")
    print(f"{'='*55}")

    train_loader = DataLoader(
        make_dataset(texts_np[train_idx].tolist(), labels_np[train_idx].tolist(), severities_np[train_idx].tolist()),
        batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(
        make_dataset(texts_np[val_idx].tolist(), labels_np[val_idx].tolist(), severities_np[val_idx].tolist()),
        batch_size=4, shuffle=False, collate_fn=collate_fn)

    model     = load_fresh_model()
    optimizer = torch.optim.AdamW([
        {"params": model.bert.parameters(),          "lr": LEARNING_RATE,      "weight_decay": 0.01},
        {"params": model.electra.parameters(),       "lr": LEARNING_RATE,      "weight_decay": 0.01},
        {"params": model.bert_pool.parameters(),     "lr": LEARNING_RATE * 5},
        {"params": model.electra_pool.parameters(),  "lr": LEARNING_RATE * 5},
        {"params": model.shared.parameters(),        "lr": LEARNING_RATE * 10},
        {"params": model.classifier.parameters(),    "lr": LEARNING_RATE * 10},
        {"params": model.severity_head.parameters(), "lr": LEARNING_RATE * 10},
    ])
    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer, int(0.1 * total_steps), total_steps)

    fold_best_acc, fold_best_state = 0.0, None

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            y_cls = batch["label"].to(device)
            y_sev = batch["severity"].to(device).unsqueeze(1)
            logits, sev = model(ids, mask)
            loss = loss_cls(logits, y_cls) + SEVERITY_ALPHA * loss_reg(sev, y_sev)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

        model.eval()
        preds_all, true_all = [], []
        with torch.no_grad():
            for batch in val_loader:
                logits, _ = model(batch["input_ids"].to(device),
                                   batch["attention_mask"].to(device))
                preds_all.extend(torch.argmax(logits, 1).cpu().numpy())
                true_all.extend(batch["label"].numpy())

        acc = accuracy_score(true_all, preds_all)
        f1  = f1_score(true_all, preds_all, average="weighted", zero_division=0)
        print(f"  Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | "
              f"Acc: {acc:.4f} | F1: {f1:.4f}")

        if acc > fold_best_acc:
            fold_best_acc   = acc
            fold_best_state = {k: v.clone() for k, v in model.state_dict().items()}

    all_accs.append(fold_best_acc)
    all_f1s.append(f1)
    print(f"  → Fold Best: {fold_best_acc:.4f}")
    if fold_best_acc > best_val_acc:
        best_val_acc, best_model_state = fold_best_acc, fold_best_state

print(f"\n{'='*55}")
print(f"  CV Accuracy : {np.mean(all_accs):.4f} ± {np.std(all_accs):.4f}")
print(f"  CV F1       : {np.mean(all_f1s):.4f}")
print(f"  Best Fold   : {max(all_accs):.4f}")

# ===============================
# 13║ Save Best Model
# ===============================
bert_inf    = AutoModel.from_pretrained("bert-base-uncased").to(device)
electra_inf = AutoModel.from_pretrained("google/electra-base-discriminator").to(device)
final_model = DualLinguisticModel(bert_inf, electra_inf).to(device)
final_model.load_state_dict(best_model_state)
final_model.eval()

save_path = os.path.join(BASE_PATH, "schizophrenia_severity_model.pt")
torch.save(best_model_state, save_path)
print(f"\n✓ Model saved → {save_path}")

# ===============================
# 14║ Inference
# ===============================
SEVERITY_BANDS = [
    (0.00, 0.20, "Minimal / None"),
    (0.20, 0.40, "Mild"),
    (0.40, 0.60, "Moderate"),
    (0.60, 0.80, "Moderately Severe"),
    (0.80, 1.01, "Severe"),
]

def severity_label(s):
    for lo, hi, lbl in SEVERITY_BANDS:
        if lo <= s < hi: return lbl
    return "Severe"

def predict_from_text(text):
    final_model.eval()
    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    padding="max_length", max_length=MAX_LENGTH)
    with torch.no_grad():
        logits, sev = final_model(enc["input_ids"].to(device),
                                   enc["attention_mask"].to(device))
    probs = torch.softmax(logits, 1).cpu().numpy()[0]
    lbl   = int(torch.argmax(logits, 1))
    s     = float(sev.item())
    return {
        "prediction":     "Schizophrenia Risk" if lbl == 1 else "Control",
        "confidence":     f"{probs[lbl]*100:.1f}%",
        "severity_score": f"{s*100:.1f} / 100",
        "severity_label": severity_label(s),
    }

def predict_from_audio(audio_path):
    compute = "float16" if device.type == "cuda" else "int8"
    hw      = "cuda"   if device.type == "cuda" else "cpu"
    wm = WhisperModel("medium", device=hw, compute_type=compute)
    segments, _ = wm.transcribe(audio_path, language="en", beam_size=5, vad_filter=True)
    transcript = " ".join(s.text for s in segments).strip()
    del wm; torch.cuda.empty_cache()
    result = predict_from_text(transcript)
    result["transcript"] = transcript
    return result

# ---- Quick test ----
print("\n--- Inference Test ---")
sample = ("I feel disconnected from reality. People seem to speak about me indirectly. "
          "I cannot concentrate and my thoughts are disorganized.")
print(predict_from_text(sample))

print("\nTo run on a new audio file:")
print("  print(predict_from_audio('/path/to/audio.wav'))")
