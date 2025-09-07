import os
import sys
import glob
import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm
import importlib.util
import pandas as pd

# Add NeXt-TDNN directory to path
sys.path.append("/your/path/next_tdnn")

# ===== Paths =====
CONFIG_PATH  = "/your/path/next_tdnn/NeXt_TDNN_C256_B3_K65_7_cyclical_lr_step.py"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

PROTO_FILE   = "/your/path/.txt"
OUTPUT_FILE  = "/your/path/results/.txt"

# ===== 1. Load config =====
spec = importlib.util.spec_from_file_location("asv_config", CONFIG_PATH)
cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)

# ===== 2. Build model =====
fe_module = importlib.import_module('preprocessing.' + cfg.FEATURE_EXTRACTOR)
feature_extractor = fe_module.feature_extractor(**cfg.FEATURE_EXTRACTOR_CONFIG)

model_module = importlib.import_module('models.' + cfg.MODEL)
model_backbone = model_module.MainModel(**cfg.MODEL_CONFIG)

agg_module = importlib.import_module('aggregation.' + cfg.AGGREGATION)
aggregation = agg_module.Aggregation(**cfg.AGGREGATION_CONFIG)

from SpeakerNet import SpeakerNet
model = SpeakerNet(
    feature_extractor=feature_extractor,
    spec_aug=None,
    model=model_backbone,
    aggregation=aggregation,
    loss_function=None
).to(DEVICE)

# Load checkpoint
checkpoint = torch.load(cfg.TEST_CHECKPOINT, map_location=DEVICE)

if "state_dict" in checkpoint:
    state_dict = {
        k.replace("speaker_net.", ""): v
        for k, v in checkpoint["state_dict"].items()
    }
else:
    state_dict = {
        k.replace("speaker_net.", ""): v
        for k, v in checkpoint.items()
    }

model.load_state_dict(state_dict, strict=True)
model.eval()

# ===== 3. Utils =====
@torch.no_grad()
def extract_embedding(wav_path):
    wav, sr = sf.read(wav_path)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    embedding = model(wav).detach().cpu().numpy().flatten()
    return embedding

def cosine(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denom == 0 else float(np.dot(a, b) / denom)

# ===== 4. Load protocol =====
proto_df = pd.read_csv(PROTO_FILE, sep=" ", header=None)
proto_df.columns = ["spk_id", "file_id", "label1", "label2"]

# ===== 5. Embedding cache =====
embeddings = {}

# ===== 6. Compute scores =====
results = []
for idx, row in tqdm(proto_df.iterrows(), total=len(proto_df), desc="Scoring nontarget"):
    spk_id, file_id, label1, label2 = row

    # Find enrollment file
    enroll_pattern = os.path.join("/your/path/enr_audio/eval", f"{spk_id}_*.flac")
    enroll_files = glob.glob(enroll_pattern)
    if not enroll_files:
        continue
    enroll_fp = enroll_files[0]

    # Find test file (file_id may have multiple extensions)
    test_fp = None
    for ext in (".wav", ".flac", ".mp3", ".m4a", ".ogg"):
        cand = os.path.join("/your/path/ASVspoof2019_LA_eval/flac", file_id + ext)
        if os.path.exists(cand):
            test_fp = cand
            break
    if test_fp is None:
        continue

    # Extract and cache embeddings
    if enroll_fp not in embeddings:
        embeddings[enroll_fp] = extract_embedding(enroll_fp)
    if test_fp not in embeddings:
        embeddings[test_fp] = extract_embedding(test_fp)

    emb1 = embeddings[enroll_fp]
    emb2 = embeddings[test_fp]

    score = cosine(emb1, emb2)
    results.append((test_fp, enroll_fp, score, spk_id, file_id, label1, label2))

# ===== 7. Save results =====
with open(OUTPUT_FILE, "w", encoding="utf-8") as fw:
    fw.write("file,enroll_file,cosine_score,spk_id,file_id,label1,label2\n")
    for f1, enr, s, spk, fid, l1, l2 in results:
        fw.write(f"{f1},{enr},{s:.6f},{spk},{fid},{l1},{l2}\n")

print(f"\n✅ Done. Saved {len(results)} scores to {OUTPUT_FILE}")
