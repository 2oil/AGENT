# VoxCeleb EER 
# import os
# import torch
# import numpy as np
# from tqdm import tqdm
# import soundfile as sf
# from sklearn.metrics import roc_curve
# import sys

# # 모델 import 경로 추가 (필요 시 조정)
# sys.path.append("/home/eoil/AGENT/")  # src 기준
# from ResNetModels.ResNetSE34V2 import MainModel

# # 설정
# MODEL_PATH = "./ResNetModels/baseline_v2_ap.model"  # 절대경로 가능
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# WAV_ROOT = "/home/eoil/AGENT/VoxCeleb/vox1_test_wav/wav"
# PROTOCOL_FILE = "/home/eoil/AGENT/VoxCeleb/veri_test2.txt"

# # ✅ 모델 초기화
# model = MainModel().to(DEVICE)

# # ✅ 체크포인트 로딩
# checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
# state_dict = checkpoint.get("model", checkpoint)

# # ✅ "__S__." 또는 "__L__." prefix 제거
# new_state_dict = {}
# for k, v in state_dict.items():
#     if k.startswith("__S__."):
#         new_key = k.replace("__S__.", "")
#     elif k.startswith("__L__."):
#         continue  # Loss 관련 키는 모델에 필요 없음
#     else:
#         new_key = k
#     new_state_dict[new_key] = v

# # ✅ 로딩
# model.load_state_dict(new_state_dict, strict=True)
# model.eval()


# # ✅ 임베딩 추출 함수
# def extract_embedding(wav_path):
#     wav, sr = sf.read(wav_path)
#     wav_tensor = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(DEVICE)
#     with torch.no_grad():
#         embedding = model(wav_tensor).squeeze().cpu().numpy()
#     return embedding

# # ✅ 평가 루프
# embeddings = {}
# scores, labels = [], []

# with open(PROTOCOL_FILE, "r") as f:
#     for line in tqdm(f.readlines(), desc="Evaluating ResNetSE34V2"):
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

# print(f"\n✅ ResNetSE34V2 ASV EER: {eer * 100:.2f}%")
# print(f"✅ Threshold at EER: {eer_threshold:.4f}")
###############################################################
# VoxCeleb score
# import os
# import torch
# import numpy as np
# from tqdm import tqdm
# import soundfile as sf
# import sys

# # 모델 import 경로 추가 (필요 시 조정)
# sys.path.append("/home/eoil/AGENT/")  # src 기준
# from ResNetModels.ResNetSE34V2 import MainModel

# # ===== 설정 =====
# MODEL_PATH = "./ResNetModels/baseline_v2_ap.model"  # 절대경로 가능
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ENROLL_ROOT = "/home/eoil/AGENT/VoxCeleb1/test/wav"              # ✅ enroll 오디오
# TEST_ROOT   = "/home/eoil/AGENT/adv_examples/both_agent_004"     # ✅ adversarial/test 오디오

# PROTOCOL_FILE = "/home/eoil/AGENT/VoxCeleb1/experiments/nontarget_4000_agent.txt"
# OUTPUT_FILE   = "/home/eoil/AGENT/VoxCeleb1/experiments/results/resnet34.txt"

# # ===== 모델 초기화 =====
# model = MainModel().to(DEVICE)

# # 체크포인트 로딩
# checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
# state_dict = checkpoint.get("model", checkpoint)

# # "__S__." 또는 "__L__." prefix 제거
# new_state_dict = {}
# for k, v in state_dict.items():
#     if k.startswith("__S__."):
#         new_key = k.replace("__S__.", "")
#     elif k.startswith("__L__."):
#         continue  # Loss 관련 키는 모델에 필요 없음
#     else:
#         new_key = k
#     new_state_dict[new_key] = v

# model.load_state_dict(new_state_dict, strict=True)
# model.eval()

# # ===== 임베딩 추출 함수 =====
# def extract_embedding(wav_path):
#     wav, sr = sf.read(wav_path)
#     wav_tensor = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(DEVICE)
#     with torch.no_grad():
#         embedding = model(wav_tensor).squeeze().cpu().numpy()
#     return embedding

# # ===== 평가 루프 =====
# embeddings = {}

# with open(PROTOCOL_FILE, "r") as f, open(OUTPUT_FILE, "w") as fout:
#     for line in tqdm(f.readlines(), desc="Evaluating ResNetSE34V2"):
#         label, utt1, utt2 = line.strip().split()
        
#         # enroll은 ENROLL_ROOT, test는 TEST_ROOT
#         path1 = os.path.join(ENROLL_ROOT, utt1)
#         path2 = os.path.join(TEST_ROOT, utt2)

#         if utt1 not in embeddings:
#             embeddings[utt1] = extract_embedding(path1)
#         if utt2 not in embeddings:
#             embeddings[utt2] = extract_embedding(path2)

#         emb1 = embeddings[utt1]
#         emb2 = embeddings[utt2]

#         score = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

#         # 파일에 기록 (label, utt1, utt2, score)
#         fout.write(f"{label} {utt1} {utt2} {score:.6f}\n")

# print(f"\n✅ 점수 파일 저장 완료: {OUTPUT_FILE}")


#######################################################################################################################
# ASVspoof nontarget score
# import os
# import sys
# import glob
# import torch
# import numpy as np
# import soundfile as sf
# from tqdm import tqdm

# # ===== 경로/설정 =====
# AUDIO_DIR = "/home/eoil/AGENT/adv_examples_resnet/both_agent_001"
# PROTOCOL_FILE = "/home/eoil/AGENT/LA/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.eval.gi.trl.txt"
# ENROLL_ROOT = "/home/eoil/AGENT/enr_audio/eval"
# OUTPUT_TXT = "/home/eoil/AGENT/ICASSP2026/experiments/total_Score/resnet34.txt"

# # ===== 모델 로드 =====
# sys.path.append("/home/eoil/AGENT/")
# from ResNetModels.ResNetSE34V2 import MainModel

# MODEL_PATH = "./ResNetModels/baseline_v2_ap.model"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# model = MainModel().to(DEVICE)
# checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
# state_dict = checkpoint.get("model", checkpoint)

# # "__S__." 제거, "__L__." 제외
# new_state_dict = {}
# for k, v in state_dict.items():
#     if k.startswith("__S__."):
#         new_key = k.replace("__S__.", "")
#     elif k.startswith("__L__."):
#         continue
#     else:
#         new_key = k
#     new_state_dict[new_key] = v
# model.load_state_dict(new_state_dict, strict=True)
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
#     emb = model(wav_tensor).squeeze().cpu().numpy()
#     return emb

# def cosine(a, b):
#     denom = (np.linalg.norm(a) * np.linalg.norm(b))
#     if denom == 0:
#         return 0.0
#     return float(np.dot(a, b) / denom)

# # ===== 프로토콜 파싱 =====
# nontarget_entries = []
# with open(PROTOCOL_FILE, "r") as f:
#     for line in f:
#         parts = line.strip().split()
#         if len(parts) < 4:
#             continue
#         spk_id, utt_id, label1, label2 = parts
#         # if label2 == "nontarget":
#         nontarget_entries.append((spk_id, utt_id, label1, label2))
# # print(f"총 nontarget 엔트리 개수: {len(nontarget_entries)}")

# # ===== enroll cache =====
# enroll_cache = {}
# def find_enroll_path(spk_id):
#     if spk_id in enroll_cache:
#         return enroll_cache[spk_id]
#     pattern = os.path.join(ENROLL_ROOT, f"{spk_id}_*.flac")
#     candidates = sorted(glob.glob(pattern))
#     enroll_cache[spk_id] = candidates[0] if candidates else None
#     return enroll_cache[spk_id]

# # ===== scoring =====
# results, skipped = [], []
# embeddings, enr_embeddings = {}, {}

# for spk_id, utt_id, label1, label2 in tqdm(nontarget_entries, desc="Scoring nontarget only"):
#     test_path = os.path.join(AUDIO_DIR, f"{utt_id}.flac")
#     enroll_path = find_enroll_path(spk_id)
#     if not os.path.exists(test_path):
#         skipped.append((utt_id, "test audio not found"))
#         continue
#     if not enroll_path or not os.path.exists(enroll_path):
#         skipped.append((utt_id, f"enroll not found for {spk_id}"))
#         continue

#     try:
#         if test_path not in embeddings:
#             embeddings[test_path] = extract_embedding(test_path)
#         if enroll_path not in enr_embeddings:
#             enr_embeddings[enroll_path] = extract_embedding(enroll_path)

#         score = cosine(embeddings[test_path], enr_embeddings[enroll_path])
#         results.append((test_path, enroll_path, score, spk_id, utt_id, label1, label2))
#     except Exception as e:
#         skipped.append((utt_id, str(e)))

# # ===== 저장 =====
# os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)

# with open(OUTPUT_TXT.replace(".txt","_score.txt"), "w") as fw:
#     fw.write("file,enroll_file,cosine_score\n")
#     for f1, enr, s, *_ in results:
#         fw.write(f"{f1},{enr},{s:.6f}\n")

# with open(OUTPUT_TXT.replace(".txt","_proto.txt"), "w") as fw:
#     for _, _, _, spk_id, utt_id, label1, label2 in results:
#         fw.write(f"{spk_id} {utt_id} {label1} {label2}\n")

# print(f"\n✅ Done. Saved {len(results)} nontarget scores.")
# if skipped:
#     print(f"⚠️ Skipped {len(skipped)} entries. Examples:")
#     for p, msg in skipped[:5]:
#         print(f" - {p} :: {msg}")



#########################################################################
# LA 19로 ASV EER 구하기 #

# import os
# import sys
# import glob
# import torch
# import numpy as np
# import soundfile as sf
# from tqdm import tqdm
# from sklearn.metrics import roc_curve

# # ===== 경로/설정 =====
# EVAL_DIR = "/home/eoil/AGENT/LA/ASVspoof2019_LA_eval/flac"
# PROTOCOL_FILE = "/home/eoil/AGENT/LA/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.eval.gi.trl.txt"
# ENROLL_ROOT = "/home/eoil/AGENT/enr_audio/eval"
# OUTPUT_TXT = "/home/eoil/AGENT/results/origin/LA19_(asv)resnet.txt"

# # 모델 import 경로
# sys.path.append("/home/eoil/AGENT/")
# from ResNetModels.ResNetSE34V2 import MainModel

# MODEL_PATH = "./ResNetModels/baseline_v2_ap.model"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # ===== 모델 로드 =====
# model = MainModel().to(DEVICE)
# checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
# state_dict = checkpoint.get("model", checkpoint)
# new_state_dict = {}
# for k, v in state_dict.items():
#     if k.startswith("__S__."):
#         new_key = k.replace("__S__.", "")
#     elif k.startswith("__L__."):
#         continue
#     else:
#         new_key = k
#     new_state_dict[new_key] = v
# model.load_state_dict(new_state_dict, strict=True)
# model.eval()

# # ===== 유틸 =====
# def read_mono_float32(path):
#     wav, sr = sf.read(path, dtype="float32")
#     if wav.ndim == 2:
#         wav = wav.mean(axis=1)
#     return wav

# @torch.no_grad()
# def extract_embedding(path):
#     wav = read_mono_float32(path)
#     wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).to(DEVICE)
#     emb = model(wav_tensor).squeeze().detach().cpu().numpy()
#     return emb

# def cosine(a, b):
#     denom = (np.linalg.norm(a) * np.linalg.norm(b))
#     if denom == 0:
#         return 0.0
#     return float(np.dot(a, b) / denom)

# # ===== EVAL_DIR 인덱스: base name -> full path =====
# base2path = {}
# for fn in os.listdir(EVAL_DIR):
#     if fn.lower().endswith(".flac"):
#         base2path[os.path.splitext(fn)[0]] = os.path.join(EVAL_DIR, fn)

# # ===== enroll_id -> enroll 파일 경로 캐시 =====
# enroll_cache = {}
# def find_enroll_path(enroll_id):
#     if enroll_id in enroll_cache:
#         return enroll_cache[enroll_id]
#     pattern = os.path.join(ENROLL_ROOT, f"{enroll_id}_*.flac")
#     cands = sorted(glob.glob(pattern))
#     enroll_cache[enroll_id] = cands[0] if cands else None
#     return enroll_cache[enroll_id]

# # ===== 프로토콜 파싱 & 스코어링 (target / nontarget만) =====
# scores, labels = [], []
# results = []   # (utt_id, enroll_id, utt_path, enroll_path, score, label)
# skipped = []
# emb_cache = {}

# with open(PROTOCOL_FILE, "r") as f:
#     lines = f.readlines()

# for line in tqdm(lines, desc="Scoring target/nontarget from protocol (eval set)"):
#     parts = line.strip().split()
#     if len(parts) < 4:
#         continue

#     enroll_id, utt_id, _, trial_type = parts[0], parts[1], parts[2].lower(), parts[3].lower()

#     # target / nontarget만 사용
#     if trial_type not in ("target", "nontarget"):
#         continue

#     utt_path = base2path.get(utt_id)
#     if not utt_path or not os.path.exists(utt_path):
#         skipped.append((utt_id, "utt file not found in EVAL_DIR"))
#         continue

#     enroll_path = find_enroll_path(enroll_id)
#     if not enroll_path or not os.path.exists(enroll_path):
#         skipped.append((utt_id, f"enroll not found for {enroll_id}"))
#         continue

#     try:
#         if utt_path not in emb_cache:
#             emb_cache[utt_path] = extract_embedding(utt_path)
#         if enroll_path not in emb_cache:
#             emb_cache[enroll_path] = extract_embedding(enroll_path)

#         s = cosine(emb_cache[utt_path], emb_cache[enroll_path])
#         y = 1 if trial_type == "target" else 0

#         scores.append(s)
#         labels.append(y)
#         results.append((utt_id, enroll_id, utt_path, enroll_path, s, y))
#     except Exception as e:
#         skipped.append((utt_id, str(e)))

# # ===== EER 계산 (target=1, nontarget=0) =====
# if len(scores) == 0:
#     raise SystemExit("No valid trials to score. Check EVAL_DIR and protocol matching.")

# labels = np.asarray(labels)
# scores = np.asarray(scores)
# fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
# fnr = 1 - tpr
# idx = np.nanargmin(np.abs(fnr - fpr))
# eer = (fnr[idx] + fpr[idx]) / 2
# eer_thr = thresholds[idx]

# print(f"\n✅ #used trials: {len(scores)} | #skipped: {len(skipped)}")
# print(f"✅ ASV EER (eval target/nontarget): {eer * 100:.2f}%")
# print(f"✅ Threshold at EER: {eer_thr:.6f}")

# if skipped:
#     print("⚠️ Skipped examples (first few):")
#     for u, msg in skipped[:8]:
#         print(f" - {u}: {msg}")

# # ===== 점수 저장 (enroll_id, utt_id 구분 포함) =====
# os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)
# with open(OUTPUT_TXT, "w", encoding="utf-8") as fw:
#     fw.write("utt_id,enroll_id,file,enroll_file,cosine_score,label(target=1,nontarget=0)\n")
#     for utt_id, enroll_id, fpath, epath, s, y in results:
#         fw.write(f"{utt_id},{enroll_id},{fpath},{epath},{s:.6f},{y}\n")

# print(f"💾 Saved scores to: {OUTPUT_TXT}")

#######################################################################

# # ASVspoof 점수
import os
import sys
import glob
import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm

# ===== 경로/설정 =====
AUDIO_DIR = "/home/eoil/AGENT/adv_examples/ab/AGENT_simple/both_agent_008"
OUTPUT_TXT = "/home/eoil/AGENT/ICASSP2026/experiments/results/AB/simple/resnet34.txt"
ENROLL_ROOT = "/home/eoil/AGENT/enr_audio/eval"

# 모델 import 경로
sys.path.append("/home/eoil/AGENT/")
from ResNetModels.ResNetSE34V2 import MainModel

MODEL_PATH = "./ResNetModels/baseline_v2_ap.model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== 모델 로드 =====
model = MainModel().to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
state_dict = checkpoint.get("model", checkpoint)

# "__S__." prefix 제거, "__L__."(loss 관련) 키 제외
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

# ===== 유틸 =====
def read_mono_float32(wav_path):
    wav, sr = sf.read(wav_path, dtype="float32")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
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

# ===== enroll_id -> enroll 파일 경로 캐시 =====
enroll_cache = {}
def find_enroll_path(enroll_id):
    if enroll_id in enroll_cache:
        return enroll_cache[enroll_id]
    pattern = os.path.join(ENROLL_ROOT, f"{enroll_id}_*.flac")
    candidates = sorted(glob.glob(pattern))
    enroll_cache[enroll_id] = candidates[0] if candidates else None
    return enroll_cache[enroll_id]

# ===== adversarial 파일 전부 수집 =====
adv_files = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")))
print(f"총 adversarial 파일 개수: {len(adv_files)}")

# ===== 임베딩 캐시 =====
embeddings = {}
enr_embeddings = {}

# ===== 스코어링 =====
results = []
skipped = []

for adv_path in tqdm(adv_files, desc="Scoring all adversarial files"):
    fname = os.path.basename(adv_path)
    base = fname.replace(".wav", "")
    parts = base.split("_")

    if len(parts) < 5:
        skipped.append((fname, "Filename parsing failed"))
        continue

    utt_id = "_".join(parts[:3])     # LA_E_1006250
    enroll_id = "_".join(parts[3:])  # LA_0017

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


# ===== 결과 저장 =====
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
######################################################################################
# 프로토콜 평가
# import os
# import sys
# import glob
# import torch
# import numpy as np
# import soundfile as sf
# from tqdm import tqdm

# # ===== 경로/설정 =====
# PROTOCOL_FILE = "/home/eoil/AGENT/ICASSP2026/experiments/protocol/target_4000.txt"
# AUDIO_DIR = "/home/eoil/AGENT/LA/ASVspoof2019_LA_eval/flac"
# OUTPUT_TXT = "/home/eoil/AGENT/ICASSP2026/experiments/results/findings/noatt/noatt_target_asv.txt"
# ENROLL_ROOT = "/home/eoil/AGENT/enr_audio/eval"

# # 모델 import 경로
# sys.path.append("/home/eoil/AGENT/")
# from ResNetModels.ResNetSE34V2 import MainModel

# MODEL_PATH = "./ResNetModels/baseline_v2_ap.model"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # ===== 모델 로드 =====
# model = MainModel().to(DEVICE)
# checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
# state_dict = checkpoint.get("model", checkpoint)

# # "__S__." prefix 제거, "__L__."(loss 관련) 키 제외
# new_state_dict = {}
# for k, v in state_dict.items():
#     if k.startswith("__S__."):
#         new_key = k.replace("__S__.", "")
#     elif k.startswith("__L__."):
#         continue
#     else:
#         new_key = k
#     new_state_dict[new_key] = v

# model.load_state_dict(new_state_dict, strict=True)
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
#     emb = model(wav_tensor).squeeze().detach().cpu().numpy()
#     return emb

# def cosine(a, b):
#     denom = (np.linalg.norm(a) * np.linalg.norm(b))
#     if denom == 0:
#         return 0.0
#     return float(np.dot(a, b) / denom)

# # ===== enroll_id -> enroll 파일 경로 캐시 =====
# enroll_cache = {}
# def find_enroll_path(enroll_id):
#     if enroll_id in enroll_cache:
#         return enroll_cache[enroll_id]
#     pattern = os.path.join(ENROLL_ROOT, f"{enroll_id}_*.flac")
#     candidates = sorted(glob.glob(pattern))
#     enroll_cache[enroll_id] = candidates[0] if candidates else None
#     return enroll_cache[enroll_id]

# # ===== 프로토콜 읽기 =====
# protocol_entries = []
# with open(PROTOCOL_FILE, "r") as f:
#     for line in f:
#         parts = line.strip().split()
#         if len(parts) < 4:
#             continue
#         enroll_id, utt_id, label, trial_type = parts
#         protocol_entries.append((enroll_id, utt_id, label, trial_type))

# print(f"총 프로토콜 엔트리 개수: {len(protocol_entries)}")

# # ===== 임베딩 캐시 =====
# embeddings = {}
# enr_embeddings = {}

# # ===== 스코어링 =====
# results = []
# skipped = []

# for enroll_id, utt_id, label, trial_type in tqdm(protocol_entries, desc="Scoring from protocol"):
#     adv_path = os.path.join(AUDIO_DIR, f"{utt_id}.flac")
#     if not os.path.exists(adv_path):
#         skipped.append((utt_id, f"Adversarial audio not found: {adv_path}"))
#         continue

#     enroll_path = find_enroll_path(enroll_id)
#     if not enroll_path or not os.path.exists(enroll_path):
#         skipped.append((utt_id, f"Enroll audio not found for {enroll_id}"))
#         continue

#     try:
#         if adv_path not in embeddings:
#             embeddings[adv_path] = extract_embedding(adv_path)
#         if enroll_path not in enr_embeddings:
#             enr_embeddings[enroll_path] = extract_embedding(enroll_path)

#         s = cosine(embeddings[adv_path], enr_embeddings[enroll_path])
#         results.append((utt_id, enroll_id, s, label, trial_type))
#     except Exception as e:
#         skipped.append((utt_id, str(e)))

# # ===== 결과 저장 =====
# os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)
# with open(OUTPUT_TXT, "w", encoding="utf-8") as fw:
#     fw.write("utt_id,enroll_id,cosine_score,label,trial_type\n")
#     for utt_id, enroll_id, s, label, trial_type in results:
#         fw.write(f"{utt_id},{enroll_id},{s:.6f},{label},{trial_type}\n")

# print(f"\n✅ Done. Saved {len(results)} scores to: {OUTPUT_TXT}")
# if skipped:
#     print(f"⚠️ Skipped {len(skipped)} entries. First few issues:")
#     for p, msg in skipped[:8]:
#         print(f" - {p} :: {msg}")
