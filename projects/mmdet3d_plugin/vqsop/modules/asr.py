"""ASR module for collaborative semantic occupancy prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner.base_module import BaseModule


class ASR(BaseModule):
    """Adaptive Semantic Refinement module used after V2V feature fusion."""
    
    def __init__(self, 
                 in_channels=192, 
                 reduction_ratio=8,
                 init_cfg=None):
        super(ASR, self).__init__(init_cfg)
        
        self.in_channels = in_channels
        self.local_branch = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.GroupNorm(24, in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.context_branch = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(24, in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction_ratio, 2, kernel_size=1, bias=False)
        )
        self.refine_proj = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.GroupNorm(24, in_channels)
        )

    def forward(self, x, original_feat=None):
        identity = x
        
        feat_local = self.local_branch(x)      # (B, C, D, H, W)
        feat_context = self.context_branch(x)  # (B, C, D, H, W)
        
        combined = torch.cat([feat_local, feat_context], dim=1) # (B, 2C, D, H, W)
        
        attn_weights = self.fusion_conv(combined)
        attn_weights = F.softmax(attn_weights, dim=1)
      
        w_local = attn_weights[:, 0:1, :, :, :]   # (B, 1, D, H, W)
        w_context = attn_weights[:, 1:2, :, :, :] # (B, 1, D, H, W)
        
        fused_feat = feat_local * w_local + feat_context * w_context
        
        out = self.refine_proj(fused_feat) + identity
        
        return out
