# AGENT — A Black-box Adversarial Attack Exposing the Achilles' Heel of SASV Systems

**Repository:** Official implementation for  
**"AGENT: A Black-box Adversarial Attack Exposing the Achilles' Heel of SASV Systems"**  
Yowon Lee, Seongkyu Han, Thien-Phuc Doan, Sanghyun Hong, Souhwan Jung. (ICASSP 2026, under review)

---

## Summary

**AGENT** (Adversarial example Generation to Neutralize SASV systems) is a black-box adversarial attack designed specifically to break **SASV** (Spoofing-Aware Speaker Verification) pipelines by jointly targeting both ASV and CM objectives while avoiding mutual interference.

Key ideas:
- **Score-maximization loss** for ASV: directly maximize ASV similarity score to push examples deep into the target region (better transferability than threshold-based losses).
- **CM loss term**: keep CM score above the CM threshold to avoid being rejected by anti-spoofing countermeasures.
- **Directional-selective gradient fusion**: compute gradients of ASV and CM surrogate losses, detect misaligned directions, remove conflicting CM components, then fuse the gradients to avoid destructive interference.
- **Iterative L∞ attack (BIM)** with step size `α` and budget `ϵ` (paper used `T = 30`, `α = ϵ/10`).

---

## Repo layout (high level)

```
AGENT/
├─ attack.sh                 # launcher that picks the appropriate generator
├─ gen_ad_both.py            # AGENT (joint attack) — the one to run for AGENT
├─ gen_ad_cm.py              # CM-only attack (not needed for AGENT release usage)
├─ gen_ad_asv.py             # ASV-only attack (not needed for AGENT release usage)
├─ experiments/                     # evaluation scripts and notebooks
│  ├─ SNR.py
│  └─ experiment.ipynb
├─ fig/
│  └─ AGENT.png
├─ models/                   # model download / install helpers (readme links)
├─ eval /                    # evaluation scripts for each model
└─ README.md                 # this file
```

> **Important:** This release focuses on **AGENT**. To run AGENT, use `gen_ad_both.py` (the attack that jointly optimizes against ASV+CM). The `attack.sh` wrapper will automatically select `gen_ad_both.py` if your `adv_method` begins with `BOTH`.

---

## Requirements

- Python 3.8+  
- PyTorch (matching your CUDA)  
- soundfile, numpy, tqdm, argparse, and other standard ML/audio libs  
- Surrogate ASV/CM checkpoints (see **Prepare** below)

Recommended (paper settings):
- `T = 30` (iterations)
- `α = ϵ / 10`
- `ϵ` in `[0.001, 0.016]` depending on perceptual budget

---

## Prepare models & data

1. **Dataset (evaluation)**  
   Download the ASVspoof2019 LA evaluation set and place audio under:
   ```
   ./LA/ASVspoof2019_LA_eval/flac/
   ```

2. **Surrogate & victim models**
  ### Countermeasure (CM)
  - **AASIST** — https://github.com/clovaai/aasist.git
  - **AASIST-SSL** — https://github.com/issflab/ssl-antispoofing.git
  - **RawNet2** —  https://github.com/asvspoof-challenge/2021/blob/main/LA/Baseline-RawNet2/README.md
  - **ResNet-OC** —  https://github.com/yzyouzhang/AIR-ASVspoof.git

  ### Automatic Speaker Verification (ASV)
  - **ECAPA-TDNN** — https://github.com/TaoRuijie/ECAPA-TDNN.git
  - **ResNet34** — https://github.com/eurecom-asp/sasv-joint-optimisation.git
  - **NeXt-TDNN** — https://github.com/dmlguq456/NeXt_TDNN_ASV.git

   (Download pretrained checkpoints and put them under `models/` or update paths in `attack.sh` / `gen_ad_both.py`.)



## Quick start — run AGENT

```
bash attack.sh
```

This runs the AGENT generator with the defaults in `attack.sh`; change the command-line arguments to customize behavior.


## Evaluation (post-attack)

- Scores and metrics are saved under `./AGENT/eval/` by default.
- Example evaluation artifacts in this repo:
  - `./AGENT/eval/` — scoring scripts
  - `./AGENT/experiments/experiment.ipynb` — notebook summarizing ASR(Attack Success Rate) analyses
  - `./AGENT/experiments/SNR.py` — compute SNR of perturbations
---

## Tips & recommended settings

- AGENT is **gen_ad_both.py**-centric: do not run ASV-only or CM-only generators if your goal is the AGENT joint attack.
- Ensure surrogate models are reasonably representative of likely victim architectures (NeXt-TDNN, ECAPA, ResNet variants) to maximize transferability.
- If you increase `ϵ`, consider decreasing `α` or increasing `steps` to avoid BIM overshoot.
- Use GPUs when available—gradient-based iterative attacks are compute intensive.

---

## Reproducibility notes

- Paper experiment settings: `T=30`, `α=ϵ/10`, `ϵ` from `0.001` to `0.016`.  
- Datasets: ASVspoof2019 LA evaluation set (used for all reported metrics).  
- Check `gen_ad_both.py` defaults for exact optimizer/normalization and surrogate preprocessing used in reported experiments.

---

## Contact

For questions about the code or experiments, contact:  
**Yowon Lee** — agent251@soongsil.ac.kr
