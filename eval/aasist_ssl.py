import os
import torch
import torch.nn.functional as F
import soundfile as sf
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.aasist_ssl import Model as AasistSSL

# 경로 설정
MODEL_PATH = "./pretrained/LA_model.pth"
SSL_MODEL_PATH = "./pretrained/xlsr2_300m.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EVAL_DIR = "/home/eoil/AGENT/LA/ASVspoof2019_LA_eval/flac/"
PROTOCOL_FILE = "/home/eoil/AGENT/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"

# checkpoint 불러오기
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

# ✅ 모든 키 앞에 'module.' 붙여주기
new_state_dict = {'module.' + k if not k.startswith('module.') else k: v for k, v in state_dict.items()}

# 모델 로드
model = AasistSSL(device=DEVICE)
model = torch.nn.DataParallel(model).to(DEVICE)
model.load_state_dict(new_state_dict, strict=True)

model.eval()

# 점수 계산
scores = []
labels = []

with open(PROTOCOL_FILE, "r") as f:
    lines = f.readlines()

for line in tqdm(lines):
    parts = line.strip().split()
    utt_id, label = parts[1], parts[4]
    audio_path = os.path.join(EVAL_DIR, utt_id + ".flac")
    if not os.path.exists(audio_path):
        continue

    wav, sr = sf.read(audio_path)
    wav_tensor = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(wav_tensor)
        cm_score = logits[0, 1].cpu()

    scores.append(cm_score)
    labels.append(1 if label == "bonafide" else 0)

# EER 계산
fpr, tpr, thresholds = roc_curve(labels, scores)
fnr = 1 - tpr
eer_idx = np.nanargmin(np.absolute(fnr - fpr))
eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
eer_threshold = thresholds[eer_idx]

print(f"\n✅ EER: {eer * 100:.2f}%")
print(f"✅ Threshold at EER: {eer_threshold:.4f}")
