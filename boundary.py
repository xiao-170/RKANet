"""
Boundary modules extracted from models/model.py

This file contains the boundary compensation blocks used to enhance
and fuse boundary information for segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _select_gn_groups(c: int) -> int:
    for g in (32, 16, 8, 4, 2, 1):
        if c % g == 0:
            return g
    return 1


def ConvBNReLU(in_channel, out_channel, kernel_size=3, stride=1, groups=1):
    """
    A standard Conv2d + BatchNorm2d + ReLU6 block.
    Padding is set to keep spatial size.
    """
    padding = (kernel_size - 1) // 2
    return nn.Sequential(
        nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        ),
        nn.GroupNorm(_select_gn_groups(out_channel), out_channel),
        nn.ReLU6(inplace=True),
    )


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.
    Generates an attention map over spatial positions using avg+max pooling.
    """

    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        label = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        out = self.sigmoid(x) * label
        return out


class BoundaryCompensationEnhancementBlock(nn.Module):
    """
    Boundary Compensation Enhancement Block.
    Multi-branch convs to capture horizontal, vertical, and all-direction edges.
    """

    def __init__(self, channel: int):
        super(BoundaryCompensationEnhancementBlock, self).__init__()
        self.ch = channel

        self.branch1 = nn.Conv2d(
            self.ch, self.ch, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False
        )
        self.branch2 = nn.Conv2d(
            self.ch, self.ch, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False
        )
        self.branch3 = nn.Conv2d(
            self.ch, self.ch, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.conv1 = ConvBNReLU(self.ch, self.ch)
        self.conv2 = nn.Sequential(
            ConvBNReLU(self.ch, self.ch, kernel_size=3),
            nn.Conv2d(self.ch, self.ch, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_1 = self.branch1(x)
        x_2 = self.branch2(x)
        x_3 = self.branch3(x)
        feat_fu = self.conv1(x_1 + x_2 + x_3)
        out = feat_fu + self.conv2(feat_fu)
        return out


class BoundaryCompensationBlock(nn.Module):
    """
    Boundary Compensation Block.
    Lets higher-level features learn boundary details from lower-level ones.
    """

    def __init__(self, channel: int):
        super(BoundaryCompensationBlock, self).__init__()
        self.channel = channel

        self.sam = SpatialAttention(kernel_size=7)
        self.conv_fusion = ConvBNReLU(self.channel * 2, self.channel)

        self.mask_generation_x = nn.Sequential(
            ConvBNReLU(self.channel, self.channel),
            nn.Conv2d(self.channel, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.feature_refine = nn.Sequential(
            ConvBNReLU(self.channel, self.channel),
            nn.Conv2d(self.channel, self.channel, kernel_size=3, padding=1),
        )

        self.isRelu = nn.ReLU()
        self.isSigmoid = nn.Sigmoid()

        self.bceb = BoundaryCompensationEnhancementBlock(self.channel)

    def forward(self, d5: torch.Tensor, d4: torch.Tensor) -> torch.Tensor:
        d5 = F.interpolate(d5, d4.size()[2:], mode="bilinear", align_corners=True)
        x = self.feature_refine(d5) + d5
        x_mask = self.isSigmoid(x)
        y = self.isRelu(self.feature_refine(d4) + d4)
        boundary_att = 1 - x_mask
        be_feat = boundary_att * y
        feat = self.sam(be_feat)
        f = torch.cat([feat, x], dim=1)
        out = self.bceb(self.conv_fusion(f))
        return out


class EdgeBranch(nn.Module):
    """
    Edge extraction branch from raw image (single-scale per level, no MS fusion, no post-SA).

    Produces multi-scale edge-enhanced features aligned to U-Net encoder skips:
    - e1 at ~1/2 resolution with `c1` channels (aligns to t1)
    - e2 at ~1/4 resolution with `c2` channels (aligns to t2)
    - e3 at ~1/8 resolution with `c3` channels (aligns to t3)
    - Optional e4 at ~1/16 resolution with `c4` channels (aligns to t4)

    Each scale: MaxPool -> ConvBNReLU -> BCEB -> scale by softplus(alpha_i).
    """

    def __init__(self, in_ch: int, c1: int, c2: int, c3: int, c4: int | None = None):
        super().__init__()
        # stem to lift channels
        self.stem = ConvBNReLU(in_ch, c1, kernel_size=3)

        # scale 1: ~1/2 resolution
        self.pool1 = nn.MaxPool2d(2)
        self.conv1 = ConvBNReLU(c1, c1, kernel_size=3)
        self.edge_enhance1 = BoundaryCompensationEnhancementBlock(c1)

        # scale 2: ~1/4 resolution
        self.pool2 = nn.MaxPool2d(2)
        self.conv2 = ConvBNReLU(c1, c2, kernel_size=3)
        self.edge_enhance2 = BoundaryCompensationEnhancementBlock(c2)

        # scale 3: ~1/8 resolution
        self.pool3 = nn.MaxPool2d(2)
        self.conv3 = ConvBNReLU(c2, c3, kernel_size=3)
        self.edge_enhance3 = BoundaryCompensationEnhancementBlock(c3)

        # optional scale 4: ~1/16 resolution (aligns to t4)
        self.has_e4 = c4 is not None
        if self.has_e4:
            self.pool4 = nn.MaxPool2d(2)
            self.conv4 = ConvBNReLU(c3, c4, kernel_size=3)
            self.edge_enhance4 = BoundaryCompensationEnhancementBlock(c4)

        # learnable fusion strength for each scale
        # start with smaller contribution; model can learn to increase
        self.alpha1 = nn.Parameter(torch.tensor(0.05))
        self.alpha2 = nn.Parameter(torch.tensor(0.05))
        self.alpha3 = nn.Parameter(torch.tensor(0.05))
        if self.has_e4:
            self.alpha4 = nn.Parameter(torch.tensor(0.05))

        # note: removed post-scale SpatialAttention to keep branch lightweight

    def forward(self, x: torch.Tensor):
        """
        Compute edge features at multiple scales without cascading the
        edge-enhanced outputs downstream. This avoids leaking t3 edges
        into t4, matching the t1/t2-style independent extraction.
        """
        # stem
        x0 = self.stem(x)  # channels: c1

        # e1 @ 1/2 (from stem only)
        p1 = self.pool1(x0)  # H/2, c1
        e1 = self.edge_enhance1(self.conv1(p1))
        e1 = e1 * F.softplus(self.alpha1)

        # e2 @ 1/4 from independent downsampling path (not from e1)
        p2 = self.pool2(p1)  # H/4, c1
        c2_in = self.conv2(p2)  # -> c2
        e2 = self.edge_enhance2(c2_in)
        e2 = e2 * F.softplus(self.alpha2)

        # e3 @ 1/8 derived from further independent downsampling, not from e2
        # use c2_in as the source to satisfy conv3's expected input channels (c2)
        p3 = self.pool3(c2_in)  # H/8, c2
        c3_in = self.conv3(p3)  # -> c3
        e3 = self.edge_enhance3(c3_in)
        e3 = e3 * F.softplus(self.alpha3)

        if self.has_e4:
            # e4 @ 1/16 derived from conv path (not from e3)
            p4 = self.pool4(c3_in)  # H/16, c3
            c4_in = self.conv4(p4)  # -> c4
            e4 = self.edge_enhance4(c4_in)
            e4 = e4 * F.softplus(self.alpha4)
            return e1, e2, e3, e4

        return e1, e2, e3

    def forward_from_c1(self, c1_feat: torch.Tensor):
        """
        Reuse encoder's first-stage pre-pooling feature as the stem output.
        This skips the branch's own stem conv to reduce computation.

        Args:
            c1_feat: tensor at full resolution with channels == c1

        Returns:
            Tuple of edge features (e1, e2, e3) or (e1, e2, e3, e4)
            at 1/2, 1/4, 1/8 (and optional 1/16) resolutions.
        """
        # e1 @ 1/2
        p1 = self.pool1(c1_feat)
        e1 = self.edge_enhance1(self.conv1(p1))
        e1 = e1 * F.softplus(self.alpha1)

        # e2 @ 1/4
        p2 = self.pool2(p1)
        c2_in = self.conv2(p2)
        e2 = self.edge_enhance2(c2_in)
        e2 = e2 * F.softplus(self.alpha2)

        # e3 @ 1/8
        p3 = self.pool3(c2_in)
        c3_in = self.conv3(p3)
        e3 = self.edge_enhance3(c3_in)
        e3 = e3 * F.softplus(self.alpha3)

        if self.has_e4:
            p4 = self.pool4(c3_in)
            c4_in = self.conv4(p4)
            e4 = self.edge_enhance4(c4_in)
            e4 = e4 * F.softplus(self.alpha4)
            return e1, e2, e3, e4

        return e1, e2, e3
