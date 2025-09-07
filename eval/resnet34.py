import os
import sys
import glob
import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm

# ===== Paths & Config =====
AUDIO_DIR = "/your/path/adv_examples/"
OUTPUT_TXT = "/your/path/experiments/results/resnet34.txt"
ENROLL_ROOT = "/your/path/enr_audio/eval"

# Import model path
sys.path.append("/your/path/")
from ResNetModels.ResNetSE34V2 import MainModel

MODEL_PATH = "/your/path/ResNetModels/baseline_v2_ap.model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== Load model =====
model = MainModel().to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
state_dict = checkpoint.get("model", checkpoint)

# Remove "__S__." prefix and skip "__L__." (loss-related) keys
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("__S__."):
        new_key = k.replace("__S__.", "")
    elif k.startswith("__L__."):
        continue
    else:
        new_key = k
    new_state_dict[new_key] = v

model.load_state_dict(new_state_dict, strict=True)
model.eval()

# ===== Utils =====
def read_mono_float32(wav_path):
    wav, sr = sf.read(wav_path, dtype="float32")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)  # stereo → mono
    return wav

@torch.no_grad()
def extract_embedding(wav_path):
    wav = read_mono_float32(wav_path)
    wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).to(DEVICE)
    emb = model(wav_tensor).squeeze().detach().cpu().numpy()
    return emb

def cosine(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

# ===== Cache: enroll_id → enroll file path =====
enroll_cache = {}
def find_enroll_path(enroll_id):
    if enroll_id in enroll_cache:
        return enroll_cache[enroll_id]
    pattern = os.path.join(ENROLL_ROOT, f"{enroll_id}_*.flac")
    candidates = sorted(glob.glob(pattern))
    enroll_cache[enroll_id] = candidates[0] if candidates else None
    return enroll_cache[enroll_id]

# ===== Collect all adversarial files =====
adv_files = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")))
print(f"Total adversarial files: {len(adv_files)}")

# ===== Embedding caches =====
embeddings = {}
enr_embeddings = {}

# ===== Scoring =====
results = []
skipped = []

for adv_path in tqdm(adv_files, desc="Scoring all adversarial files"):
    fname = os.path.basename(adv_path)
    base = fname.replace(".wav", "")
    parts = base.split("_")

    if len(parts) < 5:
        skipped.append((fname, "Filename parsing failed"))
        continue

    utt_id = "_".join(parts[:3])     # e.g., LA_E_1006250
    enroll_id = "_".join(parts[3:])  # e.g., LA_0017

    enroll_path = find_enroll_path(enroll_id)
    if not enroll_path or not os.path.exists(enroll_path):
        skipped.append((fname, f"Enroll audio not found for {enroll_id}"))
        continue

    try:
        if adv_path not in embeddings:
            embeddings[adv_path] = extract_embedding(adv_path)
        if enroll_path not in enr_embeddings:
            enr_embeddings[enroll_path] = extract_embedding(enroll_path)

        s = cosine(embeddings[adv_path], enr_embeddings[enroll_path])
        results.append((utt_id, enroll_id, s))
    except Exception as e:
        skipped.append((fname, str(e)))

# ===== Save results =====
os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)
with open(OUTPUT_TXT, "w", encoding="utf-8") as fw:
    fw.write("utt_id,enroll_id,cosine_score\n")
    for utt_id, enroll_id, s in results:
        fw.write(f"{utt_id},{enroll_id},{s:.6f}\n")

print(f"\n✅ Done. Saved {len(results)} scores to: {OUTPUT_TXT}")
if skipped:
    print(f"⚠️ Skipped {len(skipped)} entries. First few issues:")
    for p, msg in skipped[:8]:
        print(f" - {p} :: {msg}")
