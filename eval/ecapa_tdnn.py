import os
import sys
import glob
import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm

# ===== Paths & Config =====
AUDIO_DIR = "/your/path/adv_examples/"
OUTPUT_TXT = "/your/path/experiments/results/ecapa_tdnn.txt"
ENROLL_ROOT = "/your/path/enr_audio/eval"

# Import path for models
sys.path.append("/your/path/")  # relative to src directory
from src.models.ecapa_tdnn import ECAPA_TDNN  # ECAPA import

# ===== Load model =====
MODEL_PATH = "/your/path/pretrained/pretrain.model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = ECAPA_TDNN(C=1024).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

# Remove "speaker_encoder." prefix from state_dict keys if present
state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
state_dict = {
    k.replace("speaker_encoder.", ""): v
    for k, v in state_dict.items()
    if k.startswith("speaker_encoder.")
}
model.load_state_dict(state_dict, strict=True)
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
    emb = model(wav_tensor, aug=False).squeeze().detach().cpu().numpy()
    return emb

def cosine(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

# ===== Cache for enroll_id → enroll file path =====
enroll_cache = {}
def find_enroll_path(enroll_id):
    if enroll_id in enroll_cache:
        return enroll_cache[enroll_id]
    pattern = os.path.join(ENROLL_ROOT, f"{enroll_id}_*.flac")
    candidates = sorted(glob.glob(pattern))
    enroll_cache[enroll_id] = candidates[0] if candidates else None
    return enroll_cache[enroll_id]

# ===== Embedding caches =====
embeddings = {}
enr_embeddings = {}

# ===== Score all adversarial files =====
results = []
skipped = []

wav_files = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")))

for wav_path in tqdm(wav_files, desc="Scoring adversarial audios"):
    fname = os.path.splitext(os.path.basename(wav_path))[0]  # e.g., LA_E_1008476_LA_0001

    parts = fname.split("_")
    if len(parts) < 5:
        skipped.append((fname, "Filename format unexpected"))
        continue

    utt_id = "_".join(parts[:3])      # e.g., LA_E_1008476
    enroll_id = "_".join(parts[3:5])  # e.g., LA_0001

    enroll_path = find_enroll_path(enroll_id)
    if not enroll_path or not os.path.exists(enroll_path):
        skipped.append((utt_id, f"Enroll audio not found for {enroll_id}"))
        continue

    try:
        if wav_path not in embeddings:
            embeddings[wav_path] = extract_embedding(wav_path)
        if enroll_path not in enr_embeddings:
            enr_embeddings[enroll_path] = extract_embedding(enroll_path)

        s = cosine(embeddings[wav_path], enr_embeddings[enroll_path])
        results.append((utt_id, enroll_id, s))
    except Exception as e:
        skipped.append((utt_id, str(e)))

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
