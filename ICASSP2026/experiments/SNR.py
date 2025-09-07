import os
import glob
import math
import soundfile as sf
import numpy as np
from tqdm import tqdm

# ===== 설정 =====
ADV_DIR = "/home/eoil/AGENT/adv_examples/ab/both_dd_008"  
ORIG_DIR = "/home/eoil/AGENT/LA/ASVspoof2019_LA_eval/flac"         
# 필요 시 librosa로 리샘플 (없으면 주석 처리하세요)
try:
    import librosa
    HAVE_LIBROSA = True
except Exception:
    HAVE_LIBROSA = False
    print("[경고] librosa 불러오기 실패. 샘플레이트가 다르면 스킵합니다.")

def load_audio_any(path):
    """soundfile로 로드 -> (waveform[np.float32], sr). 다채널은 모노로 평균."""
    wav, sr = sf.read(path, always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if wav.dtype != np.float32:
        wav = wav.astype(np.float32)
    return wav, sr

def match_and_resample(wav, sr_src, sr_tgt):
    """샘플레이트 다르면 librosa로 리샘플(가능 시)."""
    if sr_src == sr_tgt:
        return wav, sr_src
    if not HAVE_LIBROSA:
        raise RuntimeError(f"샘플레이트 불일치: src={sr_src}, tgt={sr_tgt} (librosa 미사용)")
    wav_rs = librosa.resample(wav, orig_sr=sr_src, target_sr=sr_tgt, res_type="kaiser_best")
    return wav_rs.astype(np.float32), sr_tgt

def align_lengths(x, y):
    """두 신호를 같은 길이로 맞춤(짧은 길이에 맞춰 자름)."""
    n = min(len(x), len(y))
    return x[:n], y[:n]

def snr_db(ref, test, eps=1e-12):
    """SNR = 10 * log10( sum(ref^2) / sum((test - ref)^2) )"""
    noise = test - ref
    p_sig = float(np.sum(ref.astype(np.float64)**2))
    p_nse = float(np.sum(noise.astype(np.float64)**2))
    return 10.0 * math.log10((p_sig + eps) / (p_nse + eps))

def main():
    # 원본 파일 인덱스 (stem -> 경로)
    orig_index = {
        os.path.splitext(os.path.basename(p))[0]: p
        for p in glob.glob(os.path.join(ORIG_DIR, "**", "*.*"), recursive=True)
    }

    adv_files = sorted(glob.glob(os.path.join(ADV_DIR, "**", "*.*"), recursive=True))

    missing = 0
    sr_mismatch_skip = 0
    snr_values = []

    for adv_path in tqdm(adv_files, desc="Computing SNR"):
        # === 파일명에서 확장자 제거 ===
        adv_stem_full = os.path.splitext(os.path.basename(adv_path))[0]

        # === 뒤에 _LA_xxxx 부분 잘라내기 ===
        if "_LA_" in adv_stem_full:
            stem = adv_stem_full.split("_LA_")[0]   # "LA_E_1008476_LA_0001" → "LA_E_1008476"
        else:
            stem = adv_stem_full

        if stem not in orig_index:
            missing += 1
            continue

        orig_path = orig_index[stem]

        try:
            adv_wav, adv_sr = load_audio_any(adv_path)
            orig_wav, orig_sr = load_audio_any(orig_path)

            try:
                adv_wav, adv_sr = match_and_resample(adv_wav, adv_sr, orig_sr)
            except RuntimeError:
                sr_mismatch_skip += 1
                continue

            orig_wav, adv_wav = align_lengths(orig_wav, adv_wav)
            if len(orig_wav) == 0:
                continue

            snr = snr_db(orig_wav, adv_wav)
            snr_values.append(snr)
            print(f"{stem}: SNR = {snr:.4f} dB")

        except Exception as e:
            print(f"[에러] {stem}: {e}")

    # 평균 SNR 계산
    if snr_values:
        avg_snr = np.mean(snr_values)
        print(f"\n총 평균 SNR = {avg_snr:.4f} dB")
    else:
        print("\nSNR 계산된 값이 없습니다.")

    print(f"총 비교쌍: {len(snr_values)} | 원본 없음: {missing} | SR 불일치 스킵: {sr_mismatch_skip}")

if __name__ == "__main__":
    main()
