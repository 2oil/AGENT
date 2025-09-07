# # All ASVspoof score

# import os
# import sys
# import glob
# import torch
# import soundfile as sf
# import numpy as np
# from tqdm import tqdm

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from src.models.raw_net2 import RawNet  # RawNet2 import

# # ===== 경로 =====
# MODEL_PATH   = "/home/eoil/AGENT/pretrained_rawnet2/pre_trained_DF_RawNet2.pth"
# DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# EVAL_DIR     = "/home/eoil/AGENT/LA/ASVspoof2019_LA_eval/flac"
# PROTO_FILE   = "/home/eoil/AGENT/LA/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.eval.gi.trl.txt"
# ENROLL_ROOT  = "/home/eoil/AGENT/enr_audio/eval"
# OUTPUT_TXT   = "/home/eoil/AGENT/ICASSP2026/experiments/results/rawnet2_score.txt"

# # ===== RawNet2 설정 =====
# model_config = {
#     "nb_samp": 64600, "first_conv": 1024, "in_channels": 1,
#     "filts": [20, [20, 20], [20, 128], [128, 128]],
#     "blocks": [2, 4], "nb_fc_node": 1024, "gru_node": 1024,
#     "nb_gru_layer": 3, "nb_classes": 2
# }

# # ===== 모델 로드 =====
# model = RawNet(d_args=model_config, device=DEVICE).to(DEVICE)
# ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
# state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
# model.load_state_dict(state_dict, strict=True)
# model.eval()

# # ===== 유틸 =====
# MAX_LEN = 64600

# def load_and_pad(path):
#     wav, sr = sf.read(path)
#     if len(wav) > MAX_LEN:
#         wav = wav[:MAX_LEN]
#     elif len(wav) < MAX_LEN:
#         repeat_factor = (MAX_LEN // len(wav)) + 1
#         wav = np.tile(wav, repeat_factor)[:MAX_LEN]
#     return torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(DEVICE)

# # ===== enroll cache =====
# enroll_cache = {}
# def find_enroll(spk_id):
#     if spk_id in enroll_cache:
#         return enroll_cache[spk_id]
#     pattern = os.path.join(ENROLL_ROOT, f"{spk_id}_*.flac")
#     candidates = sorted(glob.glob(pattern))
#     enroll_cache[spk_id] = candidates[0] if candidates else None
#     return enroll_cache[spk_id]

# # ===== scoring =====
# results, skipped = [], []

# with open(PROTO_FILE, "r") as f:
#     lines = f.readlines()

# for line in tqdm(lines, desc="Scoring"):
#     parts = line.strip().split()
#     if len(parts) < 2:
#         continue
#     spk_id, utt_id = parts[0], parts[1]

#     test_path   = os.path.join(EVAL_DIR, utt_id + ".flac")
#     enroll_path = find_enroll(spk_id)

#     if not os.path.exists(test_path):
#         skipped.append((utt_id, "test not found"))
#         continue
#     if not enroll_path or not os.path.exists(enroll_path):
#         skipped.append((utt_id, "enroll not found"))
#         continue

#     try:
#         wav_tensor = load_and_pad(test_path)
#         with torch.no_grad():
#             logits = model(wav_tensor)
#             score = float(logits[0, 1].cpu().numpy())

#         results.append((test_path, enroll_path, score))
#     except Exception as e:
#         skipped.append((utt_id, str(e)))

# # ===== 저장 =====
# os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)
# with open(OUTPUT_TXT, "w") as fw:
#     fw.write("file,enroll_file,cosine_score\n")
#     for f1, enr, s in results:
#         fw.write(f"{f1},{enr},{s:.6f}\n")

# print(f"\n✅ Done. Saved {len(results)} scores to {OUTPUT_TXT}")
# if skipped:
#     print(f"⚠️ Skipped {len(skipped)} entries, first few:")
#     for p, msg in skipped[:5]:
#         print(f" - {p} :: {msg}")


#################################################################
# ASVspoof score

import os
import sys
import torch
import soundfile as sf
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.raw_net2 import RawNet  # RawNet2 import

# ===== 경로 =====
AUDIO_DIR = "/home/eoil/AGENT/adv_examples_ECA_AA/both_agent_016"
OUTPUT_TXT = "/home/eoil/AGENT/ICASSP2026/experiments/results/AGENT_ECA_AA/016/rawnet.txt"

MODEL_PATH = "/home/eoil/AGENT/pretrained_rawnet2/pre_trained_DF_RawNet2.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== RawNet2 모델 설정 =====
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

# ===== 유틸 =====
MAX_LEN = 64600  # RawNet2 입력 길이

def load_and_pad(path):
    wav, sr = sf.read(path)
    if len(wav) > MAX_LEN:
        wav = wav[:MAX_LEN]
    elif len(wav) < MAX_LEN:
        repeat_factor = (MAX_LEN // len(wav)) + 1
        wav = np.tile(wav, repeat_factor)[:MAX_LEN]
    return torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(DEVICE)

# ===== 평가 =====
results, skipped = [], []

audio_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav") or f.endswith(".flac")]

for fname in tqdm(audio_files, desc="Scoring"):
    utt = os.path.splitext(fname)[0]  # 확장자 제거된 파일명
    test_path = os.path.join(AUDIO_DIR, fname)

    try:
        wav_tensor = load_and_pad(test_path)
        with torch.no_grad():
            logits = model(wav_tensor)
            score = float(logits[0, 1].cpu().numpy())  # bonafide logit
        results.append((utt, score))
    except Exception as e:
        skipped.append((utt, str(e)))

# ===== 저장 =====
os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)
with open(OUTPUT_TXT, "w", encoding="utf-8") as fw:
    for utt, s in results:
        fw.write(f"{utt} {s:.6f}\n")

print(f"\n✅ Done. Saved {len(results)} scores to: {OUTPUT_TXT}")
if skipped:
    print(f"⚠️ Skipped {len(skipped)} files. First few issues:")
    for p, msg in skipped[:8]:
        print(f" - {p} :: {msg}")
