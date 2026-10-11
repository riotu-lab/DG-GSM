# losses/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VGGLoss(nn.Module):
    """Sum of squared VGG19 feature differences."""

    def __init__(self, device=None,
                 layer_indices=(3, 8, 17, 26, 35)):
        super().__init__()

        # relu1_2, relu2_2, relu3_4, relu4_4, relu5_4.
        # Use the same selected layers as the original training.
        self.layer_indices = set(layer_indices)

        features = models.vgg19(pretrained=True).features
        self.vgg_layers = features[:max(layer_indices) + 1]

        if device is not None:
            self.vgg_layers = self.vgg_layers.to(device)

        self.vgg_layers.eval()
        for parameter in self.vgg_layers.parameters():
            parameter.requires_grad = False

    def forward(self, x, y):
        prediction_features = x
        target_features = y.detach()
        loss = x.new_zeros(())

        for index, layer in enumerate(self.vgg_layers):
            prediction_features = layer(prediction_features)
            target_features = layer(target_features)

            if index in self.layer_indices:
                difference = prediction_features - target_features

                # Squared L2 norm per image, averaged over the batch.
                loss = loss + difference.square().flatten(1).sum(1).mean()

        return loss

class FocalFrequencyLoss(nn.Module):
    """Jiang et al.'s FFL with default full-image spectrum weighting."""

    def __init__(self, alpha=1.0, device=None):
        super().__init__()
        self.alpha = alpha

    def forward(self, input, target):
        input_fft = torch.fft.fft2(
            input, dim=(-2, -1), norm="ortho"
        )
        target_fft = torch.fft.fft2(
            target.detach(), dim=(-2, -1), norm="ortho"
        )

        difference = input_fft - target_fft
        squared_distance = (
            difference.real.square()
            + difference.imag.square()
        )

        # Adaptive weights are detached from gradient computation.
        with torch.no_grad():
            weights = squared_distance.sqrt().pow(self.alpha)

            # Normalize separately for each image and RGB channel.
            maximum = weights.amax(
                dim=(-2, -1), keepdim=True
            )

            # Handle identical spectra without division by zero.
            denominator = torch.where(
                maximum > 0,
                maximum,
                torch.ones_like(maximum),
            )
            weights = (weights / denominator).clamp(0, 1)

        return (weights * squared_distance).mean()

class NIRLLoss(nn.Module):
    """Night Illumination Restoration Loss (NIRL)."""
    def __init__(self, lambda_vgg=0.01, lambda_ff=0.1, device=None):
        super().__init__()
        self.lambda_vgg = lambda_vgg
        self.lambda_ff = lambda_ff
        self.l1_loss = nn.L1Loss()
        self.vgg_loss = VGGLoss(device=device)
        self.ff_loss = FocalFrequencyLoss(device=device)

    def forward(self, input, target):
        l1 = self.l1_loss(input, target)
        vgg = self.vgg_loss(input, target)
        ff = self.ff_loss(input, target)
        return l1 + self.lambda_vgg * vgg + self.lambda_ff * ff

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss
