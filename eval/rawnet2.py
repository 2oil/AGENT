import os
import sys
import torch
import soundfile as sf
import numpy as np
from tqdm import tqdm

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.raw_net2 import RawNet  # RawNet2 import

# ===== Paths =====
AUDIO_DIR = "/your/path/adv_examples/"
OUTPUT_TXT = "/your/path/experiments/results/rawnet.txt"

MODEL_PATH = "/your/path/pretrained_rawnet2/pre_trained_DF_RawNet2.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== RawNet2 model config =====
model_config = {
    "nb_samp": 64600, "first_conv": 1024, "in_channels": 1,
    "filts": [20, [20, 20], [20, 128], [128, 128]],
    "blocks": [2, 4], "nb_fc_node": 1024, "gru_node": 1024,
    "nb_gru_layer": 3, "nb_classes": 2
}

model = RawNet(d_args=model_config, device=DEVICE).to(DEVICE)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
model.load_state_dict(state_dict, strict=True)
model.eval()

# ===== Utils =====
MAX_LEN = 64600  # RawNet2 input length

def load_and_pad(path):
    wav, sr = sf.read(path)
    if len(wav) > MAX_LEN:
        wav = wav[:MAX_LEN]
    elif len(wav) < MAX_LEN:
        repeat_factor = (MAX_LEN // len(wav)) + 1
        wav = np.tile(wav, repeat_factor)[:MAX_LEN]
    return torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(DEVICE)

# ===== Evaluation =====
results, skipped = [], []

audio_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav") or f.endswith(".flac")]

for fname in tqdm(audio_files, desc="Scoring"):
    utt = os.path.splitext(fname)[0]  # filename without extension
    test_path = os.path.join(AUDIO_DIR, fname)

    try:
        wav_tensor = load_and_pad(test_path)
        with torch.no_grad():
            logits = model(wav_tensor)
            score = float(logits[0, 1].cpu().numpy())  # bonafide logit
        results.append((utt, score))
    except Exception as e:
        skipped.append((utt, str(e)))

# ===== Save results =====
os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)
with open(OUTPUT_TXT, "w", encoding="utf-8") as fw:
    for utt, s in results:
        fw.write(f"{utt} {s:.6f}\n")

print(f"\n✅ Done. Saved {len(results)} scores to: {OUTPUT_TXT}")
if skipped:
    print(f"⚠️ Skipped {len(skipped)} files. First few issues:")
    for p, msg in skipped[:8]:
        print(f" - {p} :: {msg}")
