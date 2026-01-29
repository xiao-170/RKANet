import torch
from torch import nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math
from abc import ABCMeta, abstractmethod
from pdb import set_trace as st

from kan import KANLinear, KAN
from torch.nn import init
# MSAA removed
from boundary import EdgeBranch

# Import MambaBlock with error handling
try:
    from mamba_block import MambaBlock
except ImportError:
    print("Warning: MambaBlock not available")
    MambaBlock = None

__all__ = ['RKANet', 'D_ConvLayer', 'ConvLayer', 'PatchEmbed', 'DW_bn_relu', 'DWConv', 'KANBlock', 'KANLayer']

class KANLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., no_kan=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        
        grid_size=5
        spline_order=3
        scale_noise=0.1
        scale_base=1.0
        scale_spline=1.0
        base_activation=torch.nn.SiLU
        grid_eps=0.02
        grid_range=[-1, 1]

        if not no_kan:
            self.fc1 = KANLinear(
                        in_features,
                        hidden_features,
                        grid_size=grid_size,
                        spline_order=spline_order,
                        scale_noise=scale_noise,
                        scale_base=scale_base,
                        scale_spline=scale_spline,
                        base_activation=base_activation,
                        grid_eps=grid_eps,
                        grid_range=grid_range,
                    )
            self.fc2 = KANLinear(
                        hidden_features,
                        out_features,
                        grid_size=grid_size,
                        spline_order=spline_order,
                        scale_noise=scale_noise,
                        scale_base=scale_base,
                        scale_spline=scale_spline,
                        base_activation=base_activation,
                        grid_eps=grid_eps,
                        grid_range=grid_range,
                    )
            self.fc3 = KANLinear(
                        hidden_features,
                        out_features,
                        grid_size=grid_size,
                        spline_order=spline_order,
                        scale_noise=scale_noise,
                        scale_base=scale_base,
                        scale_spline=scale_spline,
                        base_activation=base_activation,
                        grid_eps=grid_eps,
                        grid_range=grid_range,
                    )
            # # TODO   
            # self.fc4 = KANLinear(
            #             hidden_features,
            #             out_features,
            #             grid_size=grid_size,
            #             spline_order=spline_order,
            #             scale_noise=scale_noise,
            #             scale_base=scale_base,
            #             scale_spline=scale_spline,
            #             base_activation=base_activation,
            #             grid_eps=grid_eps,
            #             grid_range=grid_range,
            #         )   

        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.fc3 = nn.Linear(hidden_features, out_features)

        # TODO
        # self.fc1 = nn.Linear(in_features, hidden_features)


        self.dwconv_1 = DW_bn_relu(hidden_features)
        self.dwconv_2 = DW_bn_relu(hidden_features)
        self.dwconv_3 = DW_bn_relu(hidden_features)

        # # TODO
        # self.dwconv_4 = DW_bn_relu(hidden_features)
    
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    

    def forward(self, x, H, W):
        # pdb.set_trace()
        B, N, C = x.shape

        x = self.fc1(x.reshape(B*N,C))
        x = x.reshape(B,N,C).contiguous()
        x = self.dwconv_1(x, H, W)
        x = self.fc2(x.reshape(B*N,C))
        x = x.reshape(B,N,C).contiguous()
        x = self.dwconv_2(x, H, W)
        x = self.fc3(x.reshape(B*N,C))
        x = x.reshape(B,N,C).contiguous()
        x = self.dwconv_3(x, H, W)

        # # TODO
        # x = x.reshape(B,N,C).contiguous()
        # x = self.dwconv_4(x, H, W)
    
        return x

class KANBlock(nn.Module):
    def __init__(self, dim, drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, no_kan=False):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim)

        self.layer = KANLayer(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, no_kan=no_kan)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # Post-LN variant: residual first, then normalize
        out = x + self.drop_path(self.layer(x, H, W))
        out = self.norm2(out)
        return out


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class DW_bn_relu(nn.Module):
    def __init__(self, dim=768):
        super(DW_bn_relu, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class D_ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(D_ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class RKANet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False,
                img_size=224, patch_size=16, in_chans=3, embed_dims=[256, 320, 512],
                no_kan=False, use_mamba=True, mamba_d_state=16, drop_rate=0.,
                drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[1, 1, 1],
                use_checkpoint=False, bi_mamba=False, use_edge_branch=False,
                edge_share_enc1: bool = False,
                mamba_kan_mode: str = 'kan_first',  # 'kan_first' | 'mamba_first' | 'parallel'
                 **kwargs):
        super().__init__()
        
        # Validate Mamba parameters
        if use_mamba:
            try:
                from mamba_block import MambaBlock
            except ImportError:
                print("Warning: MambaBlock not found, disabling Mamba")
                use_mamba = False
        
        self.use_mamba = use_mamba
        self.bi_mamba = bool(bi_mamba) if use_mamba else False
        # control ordering/fusion between KAN and Mamba per stage
        allowed_modes = {'kan_first', 'mamba_first', 'parallel'}
        if mamba_kan_mode not in allowed_modes:
            print(f"[RKANet] Unknown mamba_kan_mode={mamba_kan_mode}, fallback to 'kan_first'")
            mamba_kan_mode = 'kan_first'
        self.mamba_kan_mode = mamba_kan_mode
        kan_input_dim = embed_dims[0]

        # Existing layers...
        self.encoder1 = ConvLayer(3, kan_input_dim//8)  
        self.encoder2 = ConvLayer(kan_input_dim//8, kan_input_dim//4)  
        self.encoder3 = ConvLayer(kan_input_dim//4, kan_input_dim)

        # Optional edge extraction branch from encoder stage-1 feature only
        self.use_edge_branch = bool(use_edge_branch)
        if self.use_edge_branch:
            c1 = kan_input_dim // 8
            c2 = kan_input_dim // 4
            c3 = kan_input_dim
            # Be lenient to different EdgeBranch signatures across versions
            try:
                self.edge_branch = EdgeBranch(c1=c1, c2=c2, c3=c3)
            except TypeError:
                try:
                    # some versions expect (in_ch, c1, c2, c3[, c4]) positional
                    self.edge_branch = EdgeBranch(input_channels, c1, c2, c3)
                except TypeError:
                    # or keyword name 'in_ch'
                    self.edge_branch = EdgeBranch(in_ch=input_channels, c1=c1, c2=c2, c3=c3)
            # Boundary-Guided Skip Fusion (concat-gated residual):
            # align edge feats to skip stats, build concat-gate mask, inject as residual
            self.edge_align1 = nn.Sequential(nn.Conv2d(c1, c1, 1, bias=False), nn.BatchNorm2d(c1))
            self.edge_align2 = nn.Sequential(nn.Conv2d(c2, c2, 1, bias=False), nn.BatchNorm2d(c2))
            self.edge_align3 = nn.Sequential(nn.Conv2d(c3, c3, 1, bias=False), nn.BatchNorm2d(c3))
            self.edge_gate1 = nn.Sequential(nn.Conv2d(c1 + c1, c1, 1, bias=True), nn.Sigmoid())
            self.edge_gate2 = nn.Sequential(nn.Conv2d(c2 + c2, c2, 1, bias=True), nn.Sigmoid())
            self.edge_gate3 = nn.Sequential(nn.Conv2d(c3 + c3, c3, 1, bias=True), nn.Sigmoid())
            self.edge_beta1 = nn.Parameter(torch.tensor(0.0))
            self.edge_beta2 = nn.Parameter(torch.tensor(0.0))
            self.edge_beta3 = nn.Parameter(torch.tensor(0.0))
        # Force reusing encoder stage-1 pre-pooling feature as the only edge branch input
        # (do not start a separate branch from the raw image)
        self.edge_share_enc1 = True if self.use_edge_branch else False


        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])
        self.dnorm3 = norm_layer(embed_dims[1])
        self.dnorm4 = norm_layer(embed_dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # KAN blocks
        self.block1 = nn.ModuleList([KANBlock(
            dim=embed_dims[1], drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer)])
        self.block2 = nn.ModuleList([KANBlock(
            dim=embed_dims[2], drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer)])
        self.dblock1 = nn.ModuleList([KANBlock(
            dim=embed_dims[1], drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer)])
        self.dblock2 = nn.ModuleList([KANBlock(
            dim=embed_dims[0], drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer)])

        # Mamba blocks for all KAN layers
        if use_mamba:
            # forward-direction mamba
            self.mamba1 = MambaBlock(d_model=embed_dims[1], d_state=mamba_d_state, dropout=drop_rate)
            self.mamba2 = MambaBlock(d_model=embed_dims[2], d_state=mamba_d_state, dropout=drop_rate)  
            self.mamba3 = MambaBlock(d_model=embed_dims[1], d_state=mamba_d_state, dropout=drop_rate)
            self.mamba4 = MambaBlock(d_model=embed_dims[0], d_state=mamba_d_state, dropout=drop_rate)

            # optional backward-direction mamba for bidirectional processing
            if self.bi_mamba:
                self.mamba1_bwd = MambaBlock(d_model=embed_dims[1], d_state=mamba_d_state, dropout=drop_rate)
                self.mamba2_bwd = MambaBlock(d_model=embed_dims[2], d_state=mamba_d_state, dropout=drop_rate)
                self.mamba3_bwd = MambaBlock(d_model=embed_dims[1], d_state=mamba_d_state, dropout=drop_rate)
                self.mamba4_bwd = MambaBlock(d_model=embed_dims[0], d_state=mamba_d_state, dropout=drop_rate)
            
            # Layer norms for Mamba outputs
            self.mamba_norm1 = norm_layer(embed_dims[1])
            self.mamba_norm2 = norm_layer(embed_dims[2])
            self.mamba_norm3 = norm_layer(embed_dims[1])
            self.mamba_norm4 = norm_layer(embed_dims[0])

        # Rest of the layers...
        self.patch_embed3 = PatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, 
                                    in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed4 = PatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, 
                                    in_chans=embed_dims[1], embed_dim=embed_dims[2])

        self.decoder1 = D_ConvLayer(embed_dims[2], embed_dims[1])  
        self.decoder2 = D_ConvLayer(embed_dims[1], embed_dims[0])  
        self.decoder3 = D_ConvLayer(embed_dims[0], embed_dims[0]//4) 
        self.decoder4 = D_ConvLayer(embed_dims[0]//4, embed_dims[0]//8)
        self.decoder5 = D_ConvLayer(embed_dims[0]//8, embed_dims[0]//8)

        self.final = nn.Conv2d(embed_dims[0]//8, num_classes, kernel_size=1)

        # SkipBoundaryRefiner removed

        # init params
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]

        # Edge branch features (multi-scale from input)
        # Encoder stage-1 convs (pre-pooling feature available as enc1_feat)
        enc1_feat = self.encoder1(x)

        # Edge branch features (multi-scale) from encoder stage-1 feature only
        if getattr(self, 'use_edge_branch', False):
            e1, e2, e3 = self.edge_branch.forward_from_c1(enc1_feat)
        else:
            e1 = e2 = e3 = None

        # Encoder stages
        out = F.relu(F.max_pool2d(enc1_feat, 2, 2))
        t1 = out
        if e1 is not None:
            # concat-gated residual injection (no multiplicative scaling on main path)
            e1a = self.edge_align1(e1)
            m1 = self.edge_gate1(torch.cat([t1, e1a], dim=1))
            t1 = t1 + F.softplus(self.edge_beta1) * (m1 * e1a)
        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2))
        t2 = out
        if e2 is not None:
            e2a = self.edge_align2(e2)
            m2 = self.edge_gate2(torch.cat([t2, e2a], dim=1))
            t2 = t2 + F.softplus(self.edge_beta2) * (m2 * e2a)
        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2))
        t3 = out
        if e3 is not None:
            # fuse edge at deep skip (before patch embedding)
            e3a = self.edge_align3(e3)
            m3 = self.edge_gate3(torch.cat([t3, e3a], dim=1))
            t3 = t3 + F.softplus(self.edge_beta3) * (m3 * e3a)
            out = t3

        # Stage 4 - KAN/Mamba composition 1
        out, H, W = self.patch_embed3(out)
        if self.use_mamba and self.mamba_kan_mode == 'mamba_first':
            # Mamba first (residual)
            if self.bi_mamba:
                y_f = self.mamba1(out)
                y_b = torch.flip(self.mamba1_bwd(torch.flip(out, dims=[1])), dims=[1])
                mamba_out = 0.5 * (y_f + y_b)
            else:
                mamba_out = self.mamba1(out)
            mamba_out = self.mamba_norm1(mamba_out)
            out = out + mamba_out

        if self.mamba_kan_mode == 'parallel' and self.use_mamba:
            x_in = out
            # branch KAN
            y_kan = x_in
            for blk in self.block1:
                y_kan = blk(y_kan, H, W)
            # branch Mamba (residual)
            if self.bi_mamba:
                y_f = self.mamba1(x_in)
                y_b = torch.flip(self.mamba1_bwd(torch.flip(x_in, dims=[1])), dims=[1])
                y_m = 0.5 * (y_f + y_b)
            else:
                y_m = self.mamba1(x_in)
            y_m = self.mamba_norm1(y_m)
            y_m = x_in + y_m
            # fuse
            out = 0.5 * (y_kan + y_m)
        else:
            # serial path (KAN first or Mamba first already applied)
            for blk in self.block1:
                out = blk(out, H, W)
            if self.use_mamba and self.mamba_kan_mode == 'kan_first':
                if self.bi_mamba:
                    y_f = self.mamba1(out)
                    y_b = torch.flip(self.mamba1_bwd(torch.flip(out, dims=[1])), dims=[1])
                    mamba_out = 0.5 * (y_f + y_b)
                else:
                    mamba_out = self.mamba1(out)
                mamba_out = self.mamba_norm1(mamba_out)
                out = out + mamba_out

        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        # Bottleneck - KAN/Mamba composition 2
        out, H, W = self.patch_embed4(out)
        if self.use_mamba and self.mamba_kan_mode == 'mamba_first':
            if self.bi_mamba:
                y_f = self.mamba2(out)
                y_b = torch.flip(self.mamba2_bwd(torch.flip(out, dims=[1])), dims=[1])
                mamba_out = 0.5 * (y_f + y_b)
            else:
                mamba_out = self.mamba2(out)
            mamba_out = self.mamba_norm2(mamba_out)
            out = out + mamba_out

        if self.mamba_kan_mode == 'parallel' and self.use_mamba:
            x_in = out
            y_kan = x_in
            for blk in self.block2:
                y_kan = blk(y_kan, H, W)
            if self.bi_mamba:
                y_f = self.mamba2(x_in)
                y_b = torch.flip(self.mamba2_bwd(torch.flip(x_in, dims=[1])), dims=[1])
                y_m = 0.5 * (y_f + y_b)
            else:
                y_m = self.mamba2(x_in)
            y_m = self.mamba_norm2(y_m)
            y_m = x_in + y_m
            out = 0.5 * (y_kan + y_m)
        else:
            for blk in self.block2:
                out = blk(out, H, W)
            if self.use_mamba and self.mamba_kan_mode == 'kan_first':
                if self.bi_mamba:
                    y_f = self.mamba2(out)
                    y_b = torch.flip(self.mamba2_bwd(torch.flip(out, dims=[1])), dims=[1])
                    mamba_out = 0.5 * (y_f + y_b)
                else:
                    mamba_out = self.mamba2(out)
                mamba_out = self.mamba_norm2(mamba_out)
                out = out + mamba_out

        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        bottleneck_fm = out

        # Decoder stage 1 - KAN/Mamba composition 3
        out_dec1 = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2,2), mode='bilinear', align_corners=False))
        out = torch.add(out_dec1, t4)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1,2)
        
        if self.use_mamba and self.mamba_kan_mode == 'mamba_first':
            if self.bi_mamba:
                y_f = self.mamba3(out)
                y_b = torch.flip(self.mamba3_bwd(torch.flip(out, dims=[1])), dims=[1])
                mamba_out = 0.5 * (y_f + y_b)
            else:
                mamba_out = self.mamba3(out)
            mamba_out = self.mamba_norm3(mamba_out)
            out = out + mamba_out

        if self.mamba_kan_mode == 'parallel' and self.use_mamba:
            x_in = out
            y_kan = x_in
            for blk in self.dblock1:
                y_kan = blk(y_kan, H, W)
            if self.bi_mamba:
                y_f = self.mamba3(x_in)
                y_b = torch.flip(self.mamba3_bwd(torch.flip(x_in, dims=[1])), dims=[1])
                y_m = 0.5 * (y_f + y_b)
            else:
                y_m = self.mamba3(x_in)
            y_m = self.mamba_norm3(y_m)
            y_m = x_in + y_m
            out = 0.5 * (y_kan + y_m)
        else:
            for blk in self.dblock1:
                out = blk(out, H, W)
            if self.use_mamba and self.mamba_kan_mode == 'kan_first':
                if self.bi_mamba:
                    y_f = self.mamba3(out)
                    y_b = torch.flip(self.mamba3_bwd(torch.flip(out, dims=[1])), dims=[1])
                    mamba_out = 0.5 * (y_f + y_b)
                else:
                    mamba_out = self.mamba3(out)
                mamba_out = self.mamba_norm3(mamba_out)
                out = out + mamba_out

        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        # Decoder stage 2 - KAN/Mamba composition 4
        out_dec2 = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2,2), mode='bilinear', align_corners=False))
        out = torch.add(out_dec2, t3)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1,2)
        
        if self.use_mamba and self.mamba_kan_mode == 'mamba_first':
            if self.bi_mamba:
                y_f = self.mamba4(out)
                y_b = torch.flip(self.mamba4_bwd(torch.flip(out, dims=[1])), dims=[1])
                mamba_out = 0.5 * (y_f + y_b)
            else:
                mamba_out = self.mamba4(out)
            mamba_out = self.mamba_norm4(mamba_out)
            out = out + mamba_out

        if self.mamba_kan_mode == 'parallel' and self.use_mamba:
            x_in = out
            y_kan = x_in
            for blk in self.dblock2:
                y_kan = blk(y_kan, H, W)
            if self.bi_mamba:
                y_f = self.mamba4(x_in)
                y_b = torch.flip(self.mamba4_bwd(torch.flip(x_in, dims=[1])), dims=[1])
                y_m = 0.5 * (y_f + y_b)
            else:
                y_m = self.mamba4(x_in)
            y_m = self.mamba_norm4(y_m)
            y_m = x_in + y_m
            out = 0.5 * (y_kan + y_m)
        else:
            for blk in self.dblock2:
                out = blk(out, H, W)
            if self.use_mamba and self.mamba_kan_mode == 'kan_first':
                if self.bi_mamba:
                    y_f = self.mamba4(out)
                    y_b = torch.flip(self.mamba4_bwd(torch.flip(out, dims=[1])), dims=[1])
                    mamba_out = 0.5 * (y_f + y_b)
                else:
                    mamba_out = self.mamba4(out)
                mamba_out = self.mamba_norm4(mamba_out)
                out = out + mamba_out

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # Final decoder stages
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2,2), mode='bilinear', align_corners=False))
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2,2), mode='bilinear', align_corners=False))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2,2), mode='bilinear', align_corners=False))

        return self.final(out)
