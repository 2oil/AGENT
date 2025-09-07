import torch
import torch.nn as nn
import numpy as np
from ..attack import Attack

class CM_BIM(Attack):
    def __init__(self, model, device, eps=0.007, alpha=0.001, steps=10):
        super().__init__("BIM", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.device = device
        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        ori_images = images.clone().detach()

        self.model.eval()
        with torch.no_grad():
            before_attack = self.model(images)
            before_score = before_attack[:, 1].cpu().numpy().ravel()

        cm_threshold = 1.85

        for _ in range(self.steps):
            images.requires_grad = True
            outputs = self.model(images)[0, 1]
            loss = - torch.abs(cm_threshold - outputs)
            grad = torch.autograd.grad(loss, images, retain_graph=False, create_graph=False)[0]

            images = images + self.alpha * grad.sign()
            eta = torch.clamp(images - ori_images, min=-self.eps, max=self.eps)
            images = torch.clamp(ori_images + eta, min=-1, max=1).detach()

        with torch.no_grad():
            after_attack = self.model(images)
            after_score = after_attack[:, 1].cpu().numpy().ravel()

        return before_score, after_score, images
