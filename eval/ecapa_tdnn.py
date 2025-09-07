# import os
# import torch
# import numpy as np
# from tqdm import tqdm
# import soundfile as sf
# from sklearn.metrics import roc_curve
# import sys

# # ECAPA_TDNN 모듈 경로 추가
# sys.path.append("/home/eoil/AGENT/")  # src 디렉토리 기준
# from src.models.ecapa_tdnn import ECAPA_TDNN  # 경로에 맞게 조정

# # 설정
# MODEL_PATH = "./pretrained/pretrain.model"  # 또는 절대경로
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# WAV_ROOT = "/home/eoil/AGENT/VoxCeleb/vox1_test_wav/wav"
# PROTOCOL_FILE = "/home/eoil/AGENT/VoxCeleb/veri_test2.txt"

# # ✅ 모델 초기화
# model = ECAPA_TDNN(C=1024).to(DEVICE)
# checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

# # ✅ state_dict 로드 (prefix 제거)
# state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
# state_dict = {
#     k.replace("speaker_encoder.", ""): v
#     for k, v in state_dict.items()
#     if k.startswith("speaker_encoder.")
# }
# model.load_state_dict(state_dict, strict=True)
# model.eval()

# # ✅ 임베딩 함수
# def extract_embedding(wav_path):
#     wav, sr = sf.read(wav_path)
#     wav_tensor = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(DEVICE)
#     with torch.no_grad():
#         embedding = model(wav_tensor, aug=False).squeeze().cpu().numpy()
#     return embedding

# # ✅ 평가 시작
# embeddings = {}
# scores, labels = [], []

# with open(PROTOCOL_FILE, "r") as f:
#     for line in tqdm(f.readlines(), desc="Evaluating ECAPA-TDNN"):
#         label, utt1, utt2 = line.strip().split()
#         path1 = os.path.join(WAV_ROOT, utt1)
#         path2 = os.path.join(WAV_ROOT, utt2)

#         if utt1 not in embeddings:
#             embeddings[utt1] = extract_embedding(path1)
#         if utt2 not in embeddings:
#             embeddings[utt2] = extract_embedding(path2)

#         emb1 = embeddings[utt1]
#         emb2 = embeddings[utt2]

#         score = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
#         scores.append(score)
#         labels.append(int(label))

# # ✅ EER 계산
# fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
# fnr = 1 - tpr
# eer_idx = np.nanargmin(np.abs(fnr - fpr))
# eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
# eer_threshold = thresholds[eer_idx]

# print(f"\n✅ ECAPA-TDNN ASV EER: {eer * 100:.2f}%")
# print(f"✅ Threshold at EER: {eer_threshold:.4f}")


################
# import os
# import sys
# import glob
# import torch
# import numpy as np
# import soundfile as sf
# from tqdm import tqdm

# # ===== 경로/설정 =====
# AUDIO_DIR   = "/home/eoil/AGENT/LA/ASVspoof2019_LA_eval/flac"
# OUTPUT_TXT  = "/home/eoil/AGENT/results/NoAttack/ecapa.txt"
# PROTOCOL_FILE = "/home/eoil/AGENT/agent_protocol(nontarget)2.txt"
# ENROLL_ROOT   = "/home/eoil/AGENT/enr_audio/eval"

# # ===== 모델 import 경로 (src 기준) =====
# sys.path.append("/home/eoil/AGENT/")
# from src.models.ecapa_tdnn import ECAPA_TDNN

# # ECAPA 가중치 경로
# MODEL_PATH = "/home/eoil/AGENT/pretrained/pretrain.model"  # <- ECAPA 가중치 교체
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # ===== 모델 로드 =====
# model = ECAPA_TDNN(C=1024).to(DEVICE)
# ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

# raw_state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
# filtered = {
#     k.replace("speaker_encoder.", ""): v
#     for k, v in raw_state.items()
#     if isinstance(k, str) and k.startswith("speaker_encoder.")
# }

# try:
#     if len(filtered) > 0:
#         model.load_state_dict(filtered, strict=True)
#     else:
#         model.load_state_dict(raw_state, strict=True)
# except Exception as e:
#     print(f"[WARN] strict=True load failed: {e}\n       Falling back to strict=False")
#     model.load_state_dict(filtered if len(filtered) > 0 else raw_state, strict=False)

# model.eval()

# # ===== 유틸 =====
# def read_mono_float32(wav_path):
#     wav, sr = sf.read(wav_path, dtype="float32")
#     if wav.ndim == 2:
#         wav = wav.mean(axis=1)
#     return wav

# @torch.no_grad()
# def extract_embedding(wav_path):
#     wav = read_mono_float32(wav_path)
#     wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).to(DEVICE)
#     emb = model(wav_tensor, aug=False).squeeze().detach().cpu().numpy()
#     return emb

# def cosine(a, b):
#     denom = (np.linalg.norm(a) * np.linalg.norm(b))
#     return 0.0 if denom == 0 else float(np.dot(a, b) / denom)

# # ===== 프로토콜 파싱: enroll_id, utt_id 쌍 리스트 =====
# pairs = []
# with open(PROTOCOL_FILE, "r") as f:
#     for line in f:
#         parts = line.strip().split()
#         if len(parts) < 2:
#             continue
#         enroll_id, utt_id = parts[0], parts[1]
#         pairs.append((enroll_id, utt_id))

# print(f"[INFO] 프로토콜에서 불러온 pair 수 = {len(pairs)}")

# # ===== enroll_id -> enroll 파일 경로 캐시 =====
# enroll_cache = {}
# def find_enroll_path(enroll_id):
#     if enroll_id in enroll_cache:
#         return enroll_cache[enroll_id]
#     pattern = os.path.join(ENROLL_ROOT, f"{enroll_id}_*.flac")
#     candidates = sorted(glob.glob(pattern))
#     enroll_cache[enroll_id] = candidates[0] if candidates else None
#     return enroll_cache[enroll_id]

# # ===== utt_id -> 실제 파일 찾기 =====
# def find_test_file(utt_id):
#     for ext in (".wav", ".flac", ".mp3", ".m4a", ".ogg"):
#         cand = os.path.join(AUDIO_DIR, utt_id + ext)
#         if os.path.exists(cand):
#             return cand
#     return None

# # ===== 임베딩 캐시 =====
# embeddings = {}
# enr_embeddings = {}

# # ===== 스코어링 (프로토콜에 정의된 pair만) =====
# results = []
# skipped = []

# for enroll_id, utt_id in tqdm(pairs, desc="Scoring protocol pairs"):
#     test_fp = find_test_file(utt_id)
#     if not test_fp:
#         skipped.append((utt_id, "Test file not found"))
#         continue

#     enroll_path = find_enroll_path(enroll_id)
#     if not enroll_path or not os.path.exists(enroll_path):
#         skipped.append((utt_id, f"Enroll audio not found for {enroll_id}"))
#         continue

#     try:
#         if test_fp not in embeddings:
#             embeddings[test_fp] = extract_embedding(test_fp)
#         if enroll_path not in enr_embeddings:
#             enr_embeddings[enroll_path] = extract_embedding(enroll_path)

#         s = cosine(embeddings[test_fp], enr_embeddings[enroll_path])
#         results.append((test_fp, enroll_path, s, enroll_id, utt_id))
#     except Exception as e:
#         skipped.append((utt_id, str(e)))

# # ===== 저장 =====
# os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)
# with open(OUTPUT_TXT, "w", encoding="utf-8") as fw:
#     fw.write("file,enroll_file,cosine_score\n")
#     for f1, enr, s, eid, uid in results:
#         fw.write(f"{f1},{enr},{s:.6f}\n")

# print(f"\n✅ Done. Saved {len(results)} scores to: {OUTPUT_TXT}")
# if skipped:
#     print(f"⚠️ Skipped {len(skipped)} items. First few issues:")
#     for item in skipped[:8]:
#         print(" -", item)
#####################################################################
# ASVspoof 점수

import os
import sys
import glob
import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm

# ===== 경로/설정 =====
AUDIO_DIR = "/home/eoil/AGENT/adv_examples_NeXt_SSL/both_agent_016"
OUTPUT_TXT = "/home/eoil/AGENT/ICASSP2026/experiments/results/AGENT_NeXt_SSL/016/ecapa_tdnn.txt"
ENROLL_ROOT = "/home/eoil/AGENT/enr_audio/eval"

# 모델 import 경로
sys.path.append("/home/eoil/AGENT/")  # src 디렉토리 기준
from src.models.ecapa_tdnn import ECAPA_TDNN  # ECAPA import

# ===== 모델 로드 =====
MODEL_PATH = "./pretrained/pretrain.model"  # 경로 확인 필요
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = ECAPA_TDNN(C=1024).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

# state_dict에서 prefix 제거
state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
state_dict = {
    k.replace("speaker_encoder.", ""): v
    for k, v in state_dict.items()
    if k.startswith("speaker_encoder.")
}
model.load_state_dict(state_dict, strict=True)
model.eval()

# ===== 유틸 =====
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

# ===== enroll_id -> enroll 파일 경로 캐시 =====
enroll_cache = {}
def find_enroll_path(enroll_id):
    if enroll_id in enroll_cache:
        return enroll_cache[enroll_id]
    pattern = os.path.join(ENROLL_ROOT, f"{enroll_id}_*.flac")
    candidates = sorted(glob.glob(pattern))
    enroll_cache[enroll_id] = candidates[0] if candidates else None
    return enroll_cache[enroll_id]

# ===== 임베딩 캐시 =====
embeddings = {}
enr_embeddings = {}

# ===== adversarial 파일 전부 스코어링 =====
results = []
skipped = []

wav_files = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")))

for wav_path in tqdm(wav_files, desc="Scoring adversarial audios"):
    fname = os.path.splitext(os.path.basename(wav_path))[0]  # 예: LA_E_1008476_LA_0001

    parts = fname.split("_")
    if len(parts) < 5:
        skipped.append((fname, "Filename format unexpected"))
        continue

    utt_id = "_".join(parts[:3])      # LA_E_1008476
    enroll_id = "_".join(parts[3:5])  # LA_0001

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

# ===== 저장 =====
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
