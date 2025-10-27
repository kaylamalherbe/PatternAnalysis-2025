from torchvision.ops import StochasticDepth
import torch
from torch import nn, Tensor
from typing import List

class ConvNeXtBlock(nn.Module):
    """
    """
    def __init__(self, in_channels: int, stem_features: int,drop_p: float = .0):
        super().__init__()
        # Depthwise Conv2d (kernel=7, padding=3, groups=channels)
        self.dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels)
        # Layer Norm
        self.norm = nn.LayerNorm(in_channels, eps=1e-6, elementwise_affine=True)
        # 1x1 Conv (expand), GELU, 1x1 Conv (contract) -> MLP-like structure
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, 4 * in_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(4 * in_channels, in_channels, kernel_size=1)
        )
        # Drop Path (StochasticDepth)
        self.stochastic_depth = StochasticDepth(drop_p, mode='row') if drop_p > 0. else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.dw_conv(x)
        # Permute for LayerNorm
        x = x.permute(0, 2, 3, 1)   # (B, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.mlp(x)
        x = shortcut + self.stochastic_depth(x)
        return x

class Downsample(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()

            # Make them proper attributes so model.to(device) finds them
            self.norm = nn.LayerNorm(in_channels, eps=1e-6, elementwise_affine=True)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)

        def forward(self, x: Tensor) -> Tensor:
            # 1. Permute to (B, H, W, C) for LayerNorm
            x = x.permute(0, 2, 3, 1)
            # 2. Apply LayerNorm over channels
            x = self.norm(x)
            # 3. Permute back to (B, C, H, W) for Conv2d
            x = x.permute(0, 3, 1, 2)
            # 4. Apply Conv2d
            x = self.conv(x)
            return x

class ConvNext(nn.Module):
   """
   ConvNext Model adapted for grayscale images and binary classification.
   1 input channel, 2 output classes.
   """
   def __init__(self, num_channels: int, stem_features: int, num_classes: int = 2, depths=[3,3,9,1], widths=[96, 192, 384, 768],):
        super().__init__()
        # Stage Stem input 224 x 224 x 3
          # Conv 4x4 stride 4
        self.stem_conv = nn.Conv2d(num_channels, widths[0], kernel_size=4, stride=4)
          # layerNorm
        self.stem_norm = nn.LayerNorm(widths[0], eps=1e-6, elementwise_affine=True)

        # Stage 1 56x56xC1 Convnext block
        self.stage1 = self._make_layer(ConvNeXtBlock, widths[0], widths[1], depths[0], True)
        # Stage 2 28x28xC2
        self.stage2 = self._make_layer(ConvNeXtBlock, widths[1], widths[2], depths[1], True)
        # Stage 3 14x14xC3
        self.stage3 = self._make_layer(ConvNeXtBlock, widths[2], widths[3], depths[2], True)
        # Stage 4 7x7xC4
        self.stage4 = self._make_layer(ConvNeXtBlock, widths[3], widths[3], depths[3], False)


        # Head 1x1xC4 Global Pool and classification
        self.norm = nn.LayerNorm(widths[-1], eps=1e-6)
        self.linear = nn.Linear(widths[-1], num_classes)

   def _make_layer(self, block, in_channels, out_channels, num_blocks, ds=True):
        """
        Create ConvNeXt layers with downsampling with predefined number of blocks.
        """
        layers = []
        for i in range(num_blocks):
            layers.append(block(in_channels, in_channels))
        if ds:
          layers.append(Downsample(in_channels, out_channels))

        return nn.Sequential(*layers)


   def forward(self, x: Tensor) -> Tensor:
        # Stem
        x = self.stem_conv(x)
        # Permute for LayerNorm (B, C, H, W) -> (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.stem_norm(x)
        # Permute back (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # head
        gap = nn.AdaptiveAvgPool2d((1, 1))
        x = gap(x)
        x = x.flatten(1)
        x = self.norm(x)
        x = self.linear(x)
        return x