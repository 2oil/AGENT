# import os
# import torch
# import numpy as np
# from tqdm import tqdm
# import importlib.util
# import soundfile as sf
# from sklearn.metrics import roc_curve
# import sys
# sys.path.append("/home/eoil/AGENT/next_tdnn")

# CONFIG_PATH = "/home/eoil/AGENT/next_tdnn/NeXt_TDNN_C256_B3_K65_7_cyclical_lr_step.py"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # 데이터 경로
# WAV_ROOT = "/home/eoil/AGENT/VoxCeleb/vox1_test_wav/wav"
# PROTOCOL_FILE = "/home/eoil/AGENT/VoxCeleb/veri_test2.txt"

# # ✅ 1. Config 불러오기
# spec = importlib.util.spec_from_file_location("asv_config", CONFIG_PATH)
# cfg = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(cfg)

# # ✅ 2. 모델 구성
# fe_module = importlib.import_module('preprocessing.' + cfg.FEATURE_EXTRACTOR)
# feature_extractor = fe_module.feature_extractor(**cfg.FEATURE_EXTRACTOR_CONFIG)

# model_module = importlib.import_module('models.' + cfg.MODEL)
# model_backbone = model_module.MainModel(**cfg.MODEL_CONFIG)

# agg_module = importlib.import_module('aggregation.' + cfg.AGGREGATION)
# aggregation = agg_module.Aggregation(**cfg.AGGREGATION_CONFIG)

# from SpeakerNet import SpeakerNet
# model = SpeakerNet(
#     feature_extractor=feature_extractor,
#     spec_aug=None,
#     model=model_backbone,
#     aggregation=aggregation,
#     loss_function=None
# ).to(DEVICE)

# checkpoint = torch.load(cfg.TEST_CHECKPOINT, map_location=DEVICE)

# # ✅ 'state_dict' 안에서 'speaker_net.' prefix 제거
# if "state_dict" in checkpoint:
#     state_dict = {
#         k.replace("speaker_net.", ""): v
#         for k, v in checkpoint["state_dict"].items()
#     }
# else:
#     state_dict = {
#         k.replace("speaker_net.", ""): v
#         for k, v in checkpoint.items()
#     }

# model.load_state_dict(state_dict, strict=True)
# model.eval()

# # ✅ 4. Embedding 저장소
# embeddings = {}

# def extract_embedding(wav_path):
#     wav, sr = sf.read(wav_path)
#     wav = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(DEVICE)
#     with torch.no_grad():
#         embedding = model(wav).detach().cpu().numpy().flatten()
#     return embedding

# # ✅ 5. 평가
# scores = []
# labels = []

# with open(PROTOCOL_FILE, "r") as f:
#     lines = f.readlines()

# for line in tqdm(lines, desc="Evaluating ASV"):
#     label, utt1, utt2 = line.strip().split()
#     path1 = os.path.join(WAV_ROOT, utt1)
#     path2 = os.path.join(WAV_ROOT, utt2)

#     # 임베딩 캐싱
#     if utt1 not in embeddings:
#         embeddings[utt1] = extract_embedding(path1)
#     if utt2 not in embeddings:
#         embeddings[utt2] = extract_embedding(path2)

#     emb1 = embeddings[utt1]
#     emb2 = embeddings[utt2]

#     score = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
#     scores.append(score)
#     labels.append(int(label))

# # ✅ 6. EER 계산
# fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
# fnr = 1 - tpr
# eer_idx = np.nanargmin(np.abs(fnr - fpr))
# eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
# eer_threshold = thresholds[eer_idx]

# print(f"\n✅ NeXt-TDNN ASV EER: {eer * 100:.2f}%")
# print(f"✅ Threshold at EER: {eer_threshold:.4f}")

# score_log_path = "/home/eoil/AGENT/next.txt"
# with open(score_log_path, "w") as log_file:
#     log_file.write("utt1\tutt2\tlabel\tscore\n")
#     for line, score in zip(lines, scores):
#         label, utt1, utt2 = line.strip().split()
#         log_file.write(f"{utt1}\t{utt2}\t{label}\t{score:.6f}\n")

# print(f"✅ Score log saved to: {score_log_path}")
################################################################################

import os
import sys
import glob
import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm
import importlib.util
import pandas as pd

sys.path.append("/home/eoil/AGENT/next_tdnn")

# ===== 경로 =====
CONFIG_PATH  = "/home/eoil/AGENT/next_tdnn/NeXt_TDNN_C256_B3_K65_7_cyclical_lr_step.py"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

PROTO_FILE   = "/home/eoil/AGENT/agent_protocol(nontarget)2.txt"
OUTPUT_FILE  = "/home/eoil/AGENT/results/NoAttack/next_tdnn.txt"

# ===== 1. Config 불러오기 =====
spec = importlib.util.spec_from_file_location("asv_config", CONFIG_PATH)
cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)

# ===== 2. 모델 구성 =====
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

# ===== 3. 유틸 =====
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

# ===== 4. 프로토콜 로드 =====
proto_df = pd.read_csv(PROTO_FILE, sep=" ", header=None)
proto_df.columns = ["spk_id", "file_id", "label1", "label2"]

# ===== 5. 임베딩 캐시 =====
embeddings = {}

# ===== 6. 스코어 계산 =====
results = []
for idx, row in tqdm(proto_df.iterrows(), total=len(proto_df), desc="Scoring nontarget"):
    spk_id, file_id, label1, label2 = row

    # enrollment 파일 찾기
    enroll_pattern = os.path.join("/home/eoil/AGENT/enr_audio/eval", f"{spk_id}_*.flac")
    enroll_files = glob.glob(enroll_pattern)
    if not enroll_files:
        continue
    enroll_fp = enroll_files[0]

    # test 파일 찾기 (file_id는 LA_E_xxxxx 형식)
    test_fp = None
    for ext in (".wav", ".flac", ".mp3", ".m4a", ".ogg"):
        cand = os.path.join("/home/eoil/AGENT/LA/ASVspoof2019_LA_eval/flac", file_id + ext)
        if os.path.exists(cand):
            test_fp = cand
            break
    if test_fp is None:
        continue

    # 임베딩 추출/캐싱
    if enroll_fp not in embeddings:
        embeddings[enroll_fp] = extract_embedding(enroll_fp)
    if test_fp not in embeddings:
        embeddings[test_fp] = extract_embedding(test_fp)

    emb1 = embeddings[enroll_fp]
    emb2 = embeddings[test_fp]

    score = cosine(emb1, emb2)
    results.append((test_fp, enroll_fp, score, spk_id, file_id, label1, label2))

# ===== 7. 저장 =====
with open(OUTPUT_FILE, "w", encoding="utf-8") as fw:
    fw.write("file,enroll_file,cosine_score,spk_id,file_id,label1,label2\n")
    for f1, enr, s, spk, fid, l1, l2 in results:
        fw.write(f"{f1},{enr},{s:.6f},{spk},{fid},{l1},{l2}\n")

print(f"\n✅ Done. Saved {len(results)} scores to {OUTPUT_FILE}")
