# AGENT: A Black-box Adversarial Attack Exposing the Achilles' Heel of SASV Systems

This repository provides the official implementation of the adversarial attack framework proposed in the paper:

> **"AGENT: A Black-box Adversarial Attack Exposing the Achilles' Heel of SASV Systems"**  
> Yowon Lee, Seongkyu Han, Thien-Phuc Doan, and Souhwan Jung  
> [ICASSP 2026, Under Review]  
> 📄 *[PDF available upon request]*

---

## 🔥 Overview

Spoofing-Aware Speaker Verification (SASV) systems integrate Automatic Speaker Verification (ASV) and Spoofing Countermeasure (CM) modules.  
While robust against spoofing attacks, the vulnerability of these systems to **adversarial attacks** has been underexplored.

This repository contains code for generating **module-targeted adversarial examples** that attack:
- **CM module** (spoof → bonafide)
- **ASV module** (nontarget → target)
- **BOTH modules** simultaneously (new in AGENT)

---

## 🧪 Supported Attack Scenarios

| Module Targeted | Objective |
|------------------|-----------|
| CM module | Spoofed samples misclassified as bonafide |
| ASV module | Non-target samples accepted as target |
| BOTH modules | Joint optimization across CM and ASV |

---

## ▶️ Usage

We provide a unified script `attack.sh` that automatically selects the correct attack script (`gen_ad_cm.py`, `gen_ad_asv.py`, or `gen_ad_both.py`) based on the prefix of the attack method.

### 1. Prepare models and data

- **Dataset**: Download and extract the [ASVspoof2019-LA evaluation set](https://datashare.ed.ac.uk/handle/10283/3336).  
  Place the evaluation audio under:
  
./LA/ASVspoof2019_LA_eval/flac/


- **Models**
  [CM]
  - [AASIST](https://github.com/clovaai/aasist.git)
  - [AASIST-SSL](https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt)
  - [RawNet2](https://github.com/asvspoof-challenge/2021/blob/main/LA/Baseline-RawNet2/README.md)
  - [ResNet-OC](https://github.com/yzyouzhang/AIR-ASVspoof.git)
    
  [ASV]
  - [ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN.git)
  - [ResNet34](https://github.com/eurecom-asp/sasv-joint-optimisation.git)
  - [NeXt-TDNN](https://github.com/dmlguq456/NeXt_TDNN_ASV.git)


### 2. Run attack

```bash
bash attack.sh
```

### 3. Evaluation

get score ./AGENT/eval
get ASR ./AGENT/experiments/experiment.ipynb
get SNR ./AGENT/experiments/SNR.py


## 🖼️ Attack Flow

![SASV Attack Pipeline](figures/AGENT.png)

---

### 📌 Tips

- adv_method1 should begin with 'CM' or 'ASV' to trigger the appropriate attack.
- You can add more arguments (e.g., --epsilon, --steps) to attack.sh as needed.

---

## 📬 Contact

For questions, contact:  
**Yowon Lee** – agent251@soongsil.ac.kr