import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss', 'BoundaryBCEDiceLoss', 'CompositeRegionBoundaryLoss']


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss


class BoundaryBCEDiceLoss(nn.Module):
    """BCE(with logits) weighted by boundary prior + Dice.

    - Builds a soft boundary map from target via morphological gradient:
      edge = dilate(target) - erode(target)
    - Element-wise weight w = 1 + alpha * edge
    - Loss = mean(w * BCEWithLogits) + Dice (0.5 weight on BCE is not used here; w already scales BCE)
    """
    def __init__(self, alpha: float = 4.0, kernel_size: int = 3, lam: float = 1.0):
        super().__init__()
        self.alpha = float(alpha)
        self.k = int(max(1, kernel_size))
        # lambda controls the strength of boundary weighting (effective alpha = lam * alpha)
        self.lam = float(lam)

    # allow runtime scheduling
    def set_lambda(self, lam: float):
        try:
            self.lam = float(lam)
        except Exception:
            pass

    def _boundary_weight(self, target: torch.Tensor) -> torch.Tensor:
        # target: (B, 1, H, W) or (B, C, H, W) with C==1
        # ensure binary in [0,1]
        t = target.detach()
        k = self.k if self.k % 2 == 1 else self.k + 1
        pad = k // 2
        dil = F.max_pool2d(t, kernel_size=k, stride=1, padding=pad)
        ero = -F.max_pool2d(-t, kernel_size=k, stride=1, padding=pad)
        edge = (dil - ero).clamp_(0, 1)
        w = 1.0 + (self.lam * self.alpha) * edge
        return w

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        w = self._boundary_weight(target)
        # BCE with per-pixel weights
        bce = F.binary_cross_entropy_with_logits(input, target, weight=w)
        # Dice (same as BCEDiceLoss)
        smooth = 1e-5
        prob = torch.sigmoid(input)
        num = target.size(0)
        prob = prob.view(num, -1)
        tgt = target.view(num, -1)
        inter = (prob * tgt)
        dice = (2. * inter.sum(1) + smooth) / (prob.sum(1) + tgt.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return bce + dice


class CompositeRegionBoundaryLoss(nn.Module):
    """
    Composite loss: L_total = L_region + lam * L_boundary

    - Region term (coverage): BCEDiceLoss (0.5 * BCEWithLogits + Dice)
    - Boundary term: BCEWithLogits computed only on a soft boundary band
      extracted via morphological gradient: band = dilate(target) - erode(target)

    Options:
      lam: scalar weight for boundary term
      kernel_size: odd kernel for morph ops (controls band thickness)
      use_band_dice: optionally add a weighted Dice on the boundary band
    """

    def __init__(self, lam: float = 0.5, kernel_size: int = 3, use_band_dice: bool = False):
        super().__init__()
        self.lam = float(lam)
        self.k = int(max(1, kernel_size))
        self.use_band_dice = bool(use_band_dice)

    def set_lambda(self, lam: float):
        try:
            self.lam = float(lam)
        except Exception:
            pass

    def _boundary_band(self, target: torch.Tensor) -> torch.Tensor:
        # Soft boundary band via morphological gradient; target assumed in [0,1]
        t = target.detach()
        k = self.k if self.k % 2 == 1 else self.k + 1
        pad = k // 2
        dil = F.max_pool2d(t, kernel_size=k, stride=1, padding=pad)
        ero = -F.max_pool2d(-t, kernel_size=k, stride=1, padding=pad)
        band = (dil - ero).clamp_(0, 1)
        return band

    def _region_loss(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Same as BCEDiceLoss
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        prob = torch.sigmoid(input)
        num = target.size(0)
        prob = prob.view(num, -1)
        tgt = target.view(num, -1)
        inter = (prob * tgt)
        dice = (2.0 * inter.sum(1) + smooth) / (prob.sum(1) + tgt.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice

    def _boundary_bce(self, input: torch.Tensor, target: torch.Tensor, band: torch.Tensor) -> torch.Tensor:
        # Masked-mean BCE on band for scale stability
        eps = 1e-8
        per_pixel = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        w = band
        num = w.sum()
        if torch.is_tensor(num):
            num_val = num.item() if num.numel() == 1 else None
        else:
            num_val = float(num)
        if num_val is not None and num_val < eps:
            return per_pixel.mean()
        return (per_pixel * w).sum() / (num + eps)

    def _boundary_dice(self, input: torch.Tensor, target: torch.Tensor, band: torch.Tensor) -> torch.Tensor:
        # Weighted Dice restricted to boundary band
        smooth = 1e-5
        prob = torch.sigmoid(input)
        w = band
        # flatten per sample
        B = target.size(0)
        prob_f = prob.view(B, -1)
        tgt_f = target.view(B, -1)
        w_f = w.view(B, -1)
        inter = (w_f * prob_f * tgt_f).sum(1)
        denom = (w_f * prob_f).sum(1) + (w_f * tgt_f).sum(1)
        dice = (2.0 * inter + smooth) / (denom + smooth)
        return 1 - dice.mean()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        region = self._region_loss(input, target)
        band = self._boundary_band(target)
        b_bce = self._boundary_bce(input, target, band)
        if self.use_band_dice:
            b_dice = self._boundary_dice(input, target, band)
            boundary = 0.5 * b_bce + 0.5 * b_dice
        else:
            boundary = b_bce
        return region + self.lam * boundary
