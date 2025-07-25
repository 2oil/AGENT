import torch
import torch.nn as nn
import numpy as np
from ..attack import Attack

class CM_FGSM(Attack):
    def __init__(self, model, device, eps=0.007):
        super().__init__("FGSM", model)
        self.eps = eps
        self.device = device
        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        print(f"Starting FGSM attack with eps={self.eps}")
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        print(f"Input image shape: {images.shape}, Labels: {labels}")

        self.model.eval()

        with torch.no_grad():
            before_attack = self.model(images)
            before_score = before_attack[:, 1].cpu().numpy().ravel()
            print(f"Before attack score: {before_score}")

        images.requires_grad = True

        outputs = self.model(images)

        loss = nn.CrossEntropyLoss()

        cost = loss(outputs, labels)

        grad = torch.autograd.grad(cost, images, retain_graph=False, create_graph=False)[0]
        adv_images = images + self.eps * grad.sign()
        adv_images = torch.clamp(adv_images, min=-1, max=1)

        with torch.no_grad():
            after_attack = self.model(adv_images)
            after_score = after_attack[:, 1].cpu().numpy().ravel()
            print(f"After attack score: {after_score}")

        return before_score, after_score, adv_images