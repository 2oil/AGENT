import os
import glob
import math
import soundfile as sf
import numpy as np
from tqdm import tqdm

# ===== Paths =====
ADV_DIR = "/your/path/adv_examples/"
ORIG_DIR = "/your/path/orgin_samples"

# Try to import librosa (for resampling if needed)
try:
    import librosa
    HAVE_LIBROSA = True
except Exception:
    HAVE_LIBROSA = False
    print("[Warning] librosa import failed. Will skip files if sample rates differ.")


def load_audio_any(path):
    """Load audio with soundfile → returns (waveform[np.float32], sr). 
    If stereo, average to mono."""
    wav, sr = sf.read(path, always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if wav.dtype != np.float32:
        wav = wav.astype(np.float32)
    return wav, sr


def match_and_resample(wav, sr_src, sr_tgt):
    """Resample if sample rates differ (uses librosa if available)."""
    if sr_src == sr_tgt:
        return wav, sr_src
    if not HAVE_LIBROSA:
        raise RuntimeError(f"Sample rate mismatch: src={sr_src}, tgt={sr_tgt} (librosa not available)")
    wav_rs = librosa.resample(wav, orig_sr=sr_src, target_sr=sr_tgt, res_type="kaiser_best")
    return wav_rs.astype(np.float32), sr_tgt


def align_lengths(x, y):
    """Trim both signals to the same length (shortest length wins)."""
    n = min(len(x), len(y))
    return x[:n], y[:n]


def snr_db(ref, test, eps=1e-12):
    """Compute SNR = 10 * log10( sum(ref^2) / sum((test - ref)^2) )."""
    noise = test - ref
    p_sig = float(np.sum(ref.astype(np.float64) ** 2))
    p_nse = float(np.sum(noise.astype(np.float64) ** 2))
    return 10.0 * math.log10((p_sig + eps) / (p_nse + eps))


def main():
    # Index original files (filename stem → path)
    orig_index = {
        os.path.splitext(os.path.basename(p))[0]: p
        for p in glob.glob(os.path.join(ORIG_DIR, "**", "*.*"), recursive=True)
    }

    # Collect all adversarial files
    adv_files = sorted(glob.glob(os.path.join(ADV_DIR, "**", "*.*"), recursive=True))

    missing = 0
    sr_mismatch_skip = 0
    snr_values = []

    for adv_path in tqdm(adv_files, desc="Computing SNR"):
        adv_stem_full = os.path.splitext(os.path.basename(adv_path))[0]

        # Extract base stem (remove "_LA_xxxx" part if present)
        if "_LA_" in adv_stem_full:
            stem = adv_stem_full.split("_LA_")[0]  # e.g., "LA_E_1008476_LA_0001" → "LA_E_1008476"
        else:
            stem = adv_stem_full

        if stem not in orig_index:
            missing += 1
            continue

        orig_path = orig_index[stem]

        try:
            adv_wav, adv_sr = load_audio_any(adv_path)
            orig_wav, orig_sr = load_audio_any(orig_path)

            # Match sample rates
            try:
                adv_wav, adv_sr = match_and_resample(adv_wav, adv_sr, orig_sr)
            except RuntimeError:
                sr_mismatch_skip += 1
                continue

            # Align lengths
            orig_wav, adv_wav = align_lengths(orig_wav, adv_wav)
            if len(orig_wav) == 0:
                continue

            # Compute SNR
            snr = snr_db(orig_wav, adv_wav)
            snr_values.append(snr)
            print(f"{stem}: SNR = {snr:.4f} dB")

        except Exception as e:
            print(f"[Error] {stem}: {e}")

    # Report average SNR
    if snr_values:
        avg_snr = np.mean(snr_values)
        print(f"\nAverage SNR = {avg_snr:.4f} dB")
    else:
        print("\nNo SNR values computed.")

    print(f"Total comparisons: {len(snr_values)} | Missing originals: {missing} | Skipped (SR mismatch): {sr_mismatch_skip}")


if __name__ == "__main__":
    main()
