import torch
import torch.nn.functional as F
import librosa
import os

class ReLU:
    def __init__(self, asv_model, cm_model, enroll_path, input_path, device,
                 eps=0.007, alpha=0.001, steps=10):
        self.asv_model = asv_model
        self.cm_model = cm_model
        self.enroll_path = enroll_path
        self.input_path = input_path
        self.device = device
        self.eps = eps
        self.alpha = alpha
        self.steps = steps

    def _load_enroll_embedding(self, spk_id):
        from glob import glob
        enroll_file = glob(os.path.join(self.enroll_path, f"{spk_id}_*.flac"))
        if not enroll_file:
            print(f"[WARNING] No enrollment file for {spk_id}")
            return None
        audio, _ = librosa.load(enroll_file[0], sr=16000)
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.asv_model(audio_tensor).squeeze(0)
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
        ori = torch.tensor(ori_audio, dtype=torch.float32).unsqueeze(0).to(self.device)
        adv = ori.clone().detach().requires_grad_()

        # Original scores
        with torch.no_grad():
            asv_emb = self.asv_model(ori).squeeze(0)
            asv_score = F.cosine_similarity(target_emb, asv_emb, dim=0).item()

            cm_logits = self.cm_model(ori)  # [1, 2]
            cm_score_ori = cm_logits[0, 1]

        # thresholds
        cm_threshold = 1.8500
        asv_threshold = 0.3161

        for _ in range(self.steps):
            adv.requires_grad = True

            # ASV loss
            emb_asv = self.asv_model(adv).squeeze(0)
            asv_sim = F.cosine_similarity(target_emb, emb_asv, dim=0)
            asv_loss = asv_sim
            grad_asv = torch.autograd.grad(asv_loss, adv, retain_graph=True)[0]

            # CM loss
            cm_logits = self.cm_model(adv)
            cm_score = cm_logits[0, 1]
            cm_loss = torch.abs(cm_score - cm_threshold)
            grad_cm = torch.autograd.grad(cm_loss, adv, retain_graph=False)[0]

            # Normalize gradients
            g_cm = grad_cm / (torch.norm(grad_cm) + 1e-8)
            g_asv = grad_asv / (torch.norm(grad_asv) + 1e-8)

            # ReLU(cos) weighting
            dot = (g_asv * g_cm).sum()          
            w = F.relu(dot).detach()           
            g_total = g_asv + w * g_cm

            # L_inf update
            adv = adv + self.alpha * g_total.sign()
            perturbation = torch.clamp(adv - ori, min=-self.eps, max=self.eps)
            adv = torch.clamp(ori + perturbation, min=-1, max=1).detach()

        # Final scores
        with torch.no_grad():
            asv_emb_adv = self.asv_model(adv).squeeze(0)
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
