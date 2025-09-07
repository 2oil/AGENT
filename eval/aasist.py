# ASVspoof EER Score threshold

# import os
# import sys
# import yaml
# import torch
# import soundfile as sf
# import numpy as np
# from tqdm import tqdm
# from sklearn.metrics import roc_curve

# # ===== 경로 설정 =====
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from src.models.aasist import Model  # ✅ 클래스명은 Model

# MODEL_PATH = "/home/eoil/aasist/models/weights/AASIST.pth"
# CONFIG_PATH = "/home/eoil/AGENT/AASIST_conf.yaml"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# EVAL_DIR = "/home/eoil/AGENT/LA/ASVspoof2019_LA_eval/flac/"
# PROTOCOL_FILE = "/home/eoil/AGENT/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
# OUTPUT_FILE = "/home/eoil/AGENT/ICASSP2026/experiments/total_Score/aasist.txt"

# # ===== YAML 설정 로드 =====
# with open(CONFIG_PATH, "r") as f:
#     config = yaml.safe_load(f)
# model_config = config["model_config"]

# # ===== 모델 정의 및 로드 =====
# model = Model(d_args=model_config, device=DEVICE).to(DEVICE)
# state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
# model.load_state_dict(state_dict, strict=True)
# model = torch.nn.DataParallel(model).to(DEVICE)
# model.eval()

# # ===== 점수 계산 =====
# results = []  # (utt_id, attack_type, label, score)

# with open(PROTOCOL_FILE, "r") as f:
#     lines = f.readlines()

# for line in tqdm(lines, desc="Evaluating"):
#     parts = line.strip().split()
#     utt_id, attack_type, label = parts[1], parts[3], parts[4]
#     audio_path = os.path.join(EVAL_DIR, utt_id + ".flac")
#     if not os.path.exists(audio_path):
#         continue

#     wav, sr = sf.read(audio_path)
#     if sr != 16000:
#         continue  # AASIST는 16kHz 기준

#     # 길이 맞추기 (64600 샘플)
#     wav = wav[:64600] if len(wav) > 64600 else np.tile(wav, (64600 // len(wav) + 1))[:64600]
#     wav_tensor = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(DEVICE)

#     with torch.no_grad():
#         logits = model(wav_tensor)
#         cm_score = logits[0, 1].item()

#     results.append(f"{utt_id} {attack_type} {label} {cm_score}")

# # ===== 결과 저장 =====
# os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
# with open(OUTPUT_FILE, "w") as fw:
#     for line in results:
#         fw.write(line + "\n")

# print(f"\n✅ Saved {len(results)} scores to {OUTPUT_FILE}")


#################################################################
# ASVspoof score

# import os
# import sys
# import glob
# import yaml
# import torch
# import soundfile as sf
# import numpy as np
# from tqdm import tqdm

# # ===== 경로/설정 =====
# AUDIO_DIR = "/home/eoil/AGENT/adv_examples/cm_bim_004"
# OUTPUT_TXT = "/home/eoil/AGENT/ICASSP2026/experiments/results/findings/cm/aasist.txt"

# # src 디렉토리 경로 추가
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from src.models.aasist import Model  # 클래스명: Model

# MODEL_PATH = "/home/eoil/aasist/models/weights/AASIST.pth"
# CONFIG_PATH = "/home/eoil/AGENT/AASIST_conf.yaml"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # ===== AASIST 로드 =====
# with open(CONFIG_PATH, "r") as f:
#     config = yaml.safe_load(f)
# model_config = config["model_config"]

# model = Model(d_args=model_config, device=DEVICE).to(DEVICE)

# # state_dict 로드
# state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
# model.load_state_dict(state_dict, strict=True)

# # DataParallel (선택)
# model = torch.nn.DataParallel(model).to(DEVICE)
# model.eval()

# # ===== 유틸 =====
# def load_and_prepare(wav_path, target_len=64600, target_sr=16000):
#     """16kHz 기준 AASIST 입력 길이로 패딩/자르기. 스테레오는 모노 변환."""
#     wav, sr = sf.read(wav_path, dtype="float32")
#     if sr != target_sr:
#         raise ValueError(f"Sample rate {sr} != {target_sr}")
#     if wav.ndim == 2:  # stereo -> mono
#         wav = wav.mean(axis=1)
#     if len(wav) >= target_len:
#         wav = wav[:target_len]
#     else:
#         reps = target_len // len(wav) + 1
#         wav = np.tile(wav, reps)[:target_len]
#     return torch.from_numpy(wav).float().unsqueeze(0)  # (1, T)

# # ===== 스코어링 =====
# exts = ("*.wav", "*.flac", "*.mp3", "*.ogg", "*.m4a")
# audio_list = []
# for ext in exts:
#     audio_list.extend(glob.glob(os.path.join(AUDIO_DIR, "**", ext), recursive=True))
# audio_list = sorted(audio_list)

# if len(audio_list) == 0:
#     print(f"No audio files found under: {AUDIO_DIR}")
#     raise SystemExit(0)

# os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)
# skipped = []

# with open(OUTPUT_TXT, "w", encoding="utf-8") as fw:
#     for path in tqdm(audio_list, desc="Scoring (AASIST)"):
#         try:
#             wav_tensor = load_and_prepare(path).to(DEVICE)
#             with torch.no_grad():
#                 out = model(wav_tensor)
#                 if isinstance(out, (tuple, list)):
#                     logits = out[0]
#                 else:
#                     logits = out
#                 cm_score = logits[0, 1].item()

#             # 파일명에서 basename (확장자 제거)
#             utt_id = os.path.splitext(os.path.basename(path))[0]
#             fw.write(f"{utt_id} {cm_score:.6f}\n")

#         except Exception as e:
#             skipped.append((path, str(e)))

# print(f"\n✅ Done. Saved scores to: {OUTPUT_TXT}")
# if skipped:
#     print(f"⚠️ Skipped {len(skipped)} files. Showing first few:")
#     for p, msg in skipped[:5]:
#         print(f" - {p} :: {msg}")

#####################################################

# 프로토콜

import os
import sys
import glob
import yaml
import torch
import soundfile as sf
import numpy as np
from tqdm import tqdm

# ===== 경로/설정 =====
PROTOCOL_FILE = "/home/eoil/AGENT/ICASSP2026/experiments/protocol/spoof_4000.txt"
AUDIO_DIR = "/home/eoil/AGENT/LA/ASVspoof2019_LA_eval/flac"
OUTPUT_TXT = "/home/eoil/AGENT/ICASSP2026/experiments/results/findings/noatt/noatt_spf_cm.txt"

# src 디렉토리 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.aasist import Model  # 클래스명: Model

MODEL_PATH = "/home/eoil/aasist/models/weights/AASIST.pth"
CONFIG_PATH = "/home/eoil/AGENT/AASIST_conf.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== AASIST 로드 =====
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)
model_config = config["model_config"]

model = Model(d_args=model_config, device=DEVICE).to(DEVICE)

# state_dict 로드
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict, strict=True)

# DataParallel (선택)
model = torch.nn.DataParallel(model).to(DEVICE)
model.eval()

# ===== 유틸 =====
def load_and_prepare(wav_path, target_len=64600, target_sr=16000):
    """16kHz 기준 AASIST 입력 길이로 패딩/자르기. 스테레오는 모노 변환."""
    wav, sr = sf.read(wav_path, dtype="float32")
    if sr != target_sr:
        raise ValueError(f"Sample rate {sr} != {target_sr}")
    if wav.ndim == 2:  # stereo -> mono
        wav = wav.mean(axis=1)
    if len(wav) >= target_len:
        wav = wav[:target_len]
    else:
        reps = target_len // len(wav) + 1
        wav = np.tile(wav, reps)[:target_len]
    return torch.from_numpy(wav).float().unsqueeze(0)  # (1, T)

# ===== 프로토콜 읽기 =====
protocol_entries = []
with open(PROTOCOL_FILE, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        enroll_id, utt_id, label, trial_type = parts
        protocol_entries.append((enroll_id, utt_id, label, trial_type))

print(f"총 프로토콜 엔트리 개수: {len(protocol_entries)}")

# ===== 스코어링 =====
os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)
skipped = []
results = []

with open(OUTPUT_TXT, "w", encoding="utf-8") as fw:
    fw.write("utt_id,enroll_id,label,trial_type,cm_score\n")
    for enroll_id, utt_id, label, trial_type in tqdm(protocol_entries, desc="Scoring (AASIST)"):
        # adversarial 파일 경로 추정
        adv_path = os.path.join(AUDIO_DIR, f"{utt_id}.flac")
        if not os.path.exists(adv_path):
            skipped.append((utt_id, f"File not found: {adv_path}"))
            continue

        try:
            wav_tensor = load_and_prepare(adv_path).to(DEVICE)
            with torch.no_grad():
                out = model(wav_tensor)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                cm_score = logits[0, 1].item()

            fw.write(f"{utt_id},{enroll_id},{label},{trial_type},{cm_score:.6f}\n")
            results.append((utt_id, enroll_id, cm_score, label, trial_type))

        except Exception as e:
            skipped.append((adv_path, str(e)))

print(f"\n✅ Done. Saved {len(results)} scores to: {OUTPUT_TXT}")
if skipped:
    print(f"⚠️ Skipped {len(skipped)} files. First few issues:")
    for p, msg in skipped[:5]:
        print(f" - {p} :: {msg}")
