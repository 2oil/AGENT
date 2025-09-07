import torch
import torch.nn.functional as F
import librosa
import os

class AGENT:
    def __init__(self, asv_model, cm_model, enroll_path, input_path, device, eps=0.007, alpha=0.001, steps=10):
        self.asv_model = asv_model
        self.cm_model = cm_model
        self.enroll_path = enroll_path
        self.input_path = input_path
        self.device = device
        self.eps = eps
        self.alpha = alpha
        self.steps = steps

    def _forward_asv(self, x):
        """ASV 모델 호출 (ECAPA-TDNN은 aug 인자 필요, 다른 모델은 불필요)"""
        try:
            return self.asv_model(x, aug=False)   # ECAPA-TDNN 스타일
        except TypeError:
            return self.asv_model(x)              # RawNet, ResNet, AASIST 등

    def _load_enroll_embedding(self, spk_id):
        from glob import glob
        enroll_file = glob(os.path.join(self.enroll_path, f"{spk_id}_*.flac"))
        if not enroll_file:
            print(f"[WARNING] No enrollment file for {spk_id}")
            return None
        audio, _ = librosa.load(enroll_file[0], sr=16000)
        audio_tensor = torch.tensor(audio).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self._forward_asv(audio_tensor).squeeze(0)
        return emb

    def forward(self, target_spk, test_utterance):
        # Load audio
        test_path = os.path.join(self.input_path, f"{test_utterance}.flac")
        if not os.path.exists(test_path):
            print(f"❌ File not found: {test_path}")
            return None

        target_emb = self._load_enroll_embedding(target_spk)
        if target_emb is None:
            return None

        ori_audio, _ = librosa.load(test_path, sr=16000)
        ori = torch.tensor(ori_audio).unsqueeze(0).to(self.device)
        adv = ori.clone().detach().requires_grad_()

        # Original scores
        with torch.no_grad():
            asv_emb = self._forward_asv(ori).squeeze(0)
            asv_score = F.cosine_similarity(target_emb, asv_emb, dim=0).item()

            cm_logits = self.cm_model(ori)  # [1, 2]
            cm_score_ori = cm_logits[0, 1]

        #cm_threshold = 1.8500 
        #cm_threshold = 2.9759
        cm_threshold = 1.8503
        asv_threshold = 0.3161

        for _ in range(self.steps):
            adv.requires_grad = True

            # ASV loss
            emb_asv = self._forward_asv(adv).squeeze(0)
            asv_sim = F.cosine_similarity(target_emb, emb_asv, dim=0)
            #asv_loss = asv_sim
            asv_loss = - torch.abs(asv_threshold - asv_sim)
            grad_asv = torch.autograd.grad(asv_loss, adv, retain_graph=True)[0]

            # CM loss
            cm_logits = self.cm_model(adv)
            cm_score = cm_logits[0, 1]
            cm_loss = torch.abs(cm_threshold - cm_score)
            grad_cm = torch.autograd.grad(cm_loss, adv, retain_graph=False)[0]

            g_total = grad_asv + grad_cm

            #Normalize gradients
            #g_cm = grad_cm / (torch.norm(grad_cm) + 1e-8)
            #g_asv = grad_asv / (torch.norm(grad_asv) + 1e-8)

            #Filter opposite direction
            #dot = (g_cm * g_asv).sum()

            # if dot < 0:
            #     g_asv_filtered = g_asv - dot * g_cm
            # else:
            #     g_asv_filtered = g_asv

        #     # Combine gradients
        #     g_total = g_asv_filtered + g_cm
        #    # g_total = 0.8 * g_asv_filtered + 0.2 * g_cm
########################
            # if dot < 0:
            #      g_cm_filtered = g_cm - dot * g_asv 
            # else:
            #      g_cm_filtered = g_cm

            # g_total = g_asv + g_cm_filtered
########################           
            #g_total = g_asv + g_cm
            adv = adv + self.alpha * g_total.sign()
            perturbation = torch.clamp(adv - ori, min=-self.eps, max=self.eps)
            adv = torch.clamp(ori + perturbation, min=-1, max=1).detach()

        # Final scores
        with torch.no_grad():
            asv_emb_adv = self._forward_asv(adv).squeeze(0)
            asv_score_adv = F.cosine_similarity(target_emb, asv_emb_adv, dim=0).item()

            cm_logits_adv = self.cm_model(adv)
            cm_score_adv = cm_logits_adv[0, 1]

        return (
            cm_score_ori.cpu().item(),
            cm_score_adv.cpu().item(),
            asv_score,
            asv_score_adv,
            adv.squeeze(0).cpu().numpy()
        )

#####################################################
# VoxCeleb

# import torch
# import torch.nn.functional as F
# import librosa
# import os
# import numpy as np

# class AGENT:
#     def __init__(self, asv_model, cm_model, enroll_path, input_path, device, eps=0.007, alpha=0.001, steps=10):
#         self.asv_model = asv_model
#         self.cm_model = cm_model
#         self.enroll_path = enroll_path
#         self.input_path = input_path
#         self.device = device
#         self.eps = eps
#         self.alpha = alpha
#         self.steps = steps

#         self.enroll_cache = {}

#     def _load_enroll_embedding(self, enroll_file):
#         if enroll_file in self.enroll_cache:
#             return self.enroll_cache[enroll_file].to(self.device)

#         if not os.path.exists(enroll_file):
#             print(f"[WARNING] No enroll file: {enroll_file}")
#             return None

#         audio, _ = librosa.load(enroll_file, sr=16000)
#         audio_tensor = torch.tensor(audio).unsqueeze(0).to(self.device)
#         with torch.no_grad():
#             emb = self.asv_model(audio_tensor).squeeze(0).detach().cpu()
#         self.enroll_cache[enroll_file] = emb
#         return emb.to(self.device)


#     def forward(self, enroll_file, test_file):
#         if not os.path.exists(test_file):
#             print(f"❌ File not found: {test_file}")
#             return None

#         # Load/cached enroll embedding (CPU cache → GPU copy)
#         target_emb = self._load_enroll_embedding(enroll_file)
#         if target_emb is None:
#             return None

#         ori_audio, _ = librosa.load(test_file, sr=16000, dtype=np.float32)
#         ori = torch.tensor(ori_audio).unsqueeze(0).to(self.device)
#         adv = ori.clone().detach().requires_grad_()

#         # Original scores
#         with torch.no_grad():
#             asv_emb = self.asv_model(ori).squeeze(0)
#             asv_score = F.cosine_similarity(target_emb, asv_emb, dim=0).item()
#             cm_logits = self.cm_model(ori)
#             cm_score_ori = cm_logits[0, 1]

#         cm_threshold = 1.8500

#         for _ in range(self.steps):
#             adv = adv.detach().requires_grad_()  # detach()로 그래프 끊기

#             # ASV gradient
#             emb_asv = self.asv_model(adv).squeeze(0)
#             asv_sim = F.cosine_similarity(target_emb, emb_asv, dim=0)
#             grad_asv, = torch.autograd.grad(asv_sim, adv, retain_graph=False)

#             # CM gradient
#             cm_logits = self.cm_model(adv)
#             cm_score = cm_logits[0, 1]
#             cm_loss = -torch.abs(cm_threshold - cm_score)
#             grad_cm, = torch.autograd.grad(cm_loss, adv, retain_graph=False)

#             # Normalize
#             g_cm = grad_cm / (torch.norm(grad_cm) + 1e-8)
#             g_asv = grad_asv / (torch.norm(grad_asv) + 1e-8)

#             dot = (g_cm * g_asv).sum()
#             g_cm_filtered = g_cm - dot * g_asv if dot < 0 else g_cm
#             g_total = g_asv + g_cm_filtered

#             adv = adv + self.alpha * g_total.sign()
#             adv = torch.clamp(ori + (adv - ori).clamp(-self.eps, self.eps), -1, 1)

#         # Final scores
#         with torch.no_grad():
#             asv_emb_adv = self.asv_model(adv).squeeze(0)
#             asv_score_adv = F.cosine_similarity(target_emb, asv_emb_adv, dim=0).item()
#             cm_logits_adv = self.cm_model(adv)
#             cm_score_adv = cm_logits_adv[0, 1]

#         # adversarial 오디오를 numpy로 반환 (필요 없으면 여기서 바로 저장하는 게 더 안전)
#         adv_audio = adv.squeeze(0).detach().cpu().numpy()

#         # 불필요한 텐서 정리
#         del adv, ori, asv_emb, asv_emb_adv, grad_asv, grad_cm
#         torch.cuda.empty_cache()

#         return (
#             cm_score_ori.item(),
#             cm_score_adv.item(),
#             asv_score,
#             asv_score_adv,
#             adv_audio
#         )
