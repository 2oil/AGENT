import os
import torch
import soundfile as sf
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve
from easydict import EasyDict
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.conformer_tcm import Model as Conformer

# 경로 설정
MODEL_PATH = "/home/eoil/AGENT/best_4.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EVAL_DIR = "/home/eoil/AGENT/LA/ASVspoof2019_LA_eval/flac/"
PROTOCOL_FILE = "/home/eoil/AGENT/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"

# 모델 args 설정
args = EasyDict({
    "emb_size": 144,
    "num_encoders": 4,
    "heads": 4,
    "kernel_size": 31
})

# 모델 초기화 및 가중치 로딩
model = Conformer(args=args, device=DEVICE).to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict, strict=True)

# (필요 시 DataParallel)
# model = torch.nn.DataParallel(model).to(DEVICE)

model.eval()

# 점수 계산
scores = []
labels = []

with open(PROTOCOL_FILE, "r") as f:
    lines = f.readlines()

for line in tqdm(lines, desc="Evaluating"):
    parts = line.strip().split()
    utt_id, label = parts[1], parts[4]
    audio_path = os.path.join(EVAL_DIR, utt_id + ".flac")
    if not os.path.exists(audio_path):
        continue

    wav, sr = sf.read(audio_path)
    wav_tensor = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits, _ = model(wav_tensor)  # logits는 첫 번째 반환값
        cm_score = logits[0, 1].cpu().item()  # spoof class score

    scores.append(cm_score)
    labels.append(1 if label == "bonafide" else 0)

# EER 계산
fpr, tpr, thresholds = roc_curve(labels, scores)
fnr = 1 - tpr
eer_idx = np.nanargmin(np.abs(fnr - fpr))
eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
eer_threshold = thresholds[eer_idx]

print(f"\n✅ Conformer-TCM EER: {eer * 100:.2f}%")
print(f"✅ Threshold at EER: {eer_threshold:.4f}")
