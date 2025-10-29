from torchvision.ops import StochasticDepth
import torch
from torch import nn, Tensor
from typing import List

class ConvNeXtBlock(nn.Module):
    def __init__(self, in_channels: int, stem_features: int,  drop_p: float = .0, dropout_p: float = .1):
        super().__init__()
        # Depthwise Conv2d (kernel=7, padding=3, groups=channels)
        self.dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels)
        # Layer Norm
        self.norm = nn.LayerNorm(in_channels, eps=1e-6, elementwise_affine=True)
        # 1x1 Conv (expand), GELU, 1x1 Conv (contract) -> MLP-like structure
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, 4 * in_channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(p=dropout_p),  # Added Dropout after GELU
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
   def __init__(self, num_channels: int, stem_features: int, num_classes: int = 2, depths=[3,3,9,1], widths=[96, 192, 384, 768], dropout_p: float = 0.3, drop_path_rate: float = 0.4):
        super().__init__()
        # Stage Stem input 224 x 224 x 3
          # Conv 4x4 stride 4
        self.stem_conv = nn.Conv2d(num_channels, widths[0], kernel_size=4, stride=4)
          # layerNorm
        self.stem_norm = nn.LayerNorm(widths[0], eps=1e-6, elementwise_affine=True)

        # 1. Total number of ConvNeXt Blocks
        total_blocks = sum(depths)
        
        # 2. Generate a list of drop path rates
        # Linearly interpolate between 0.0 (or a small start rate) and the final rate (e.g., 0.5)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        
        # 3. Create stages with unique drop rates
        cur = 0 # Current index in the dpr list

        # Stage 1
        self.stage1, cur = self._make_layer(ConvNeXtBlock, widths[0], widths[1], depths[0], True, dropout_p, dpr, cur)
        # Stage 2
        self.stage2, cur = self._make_layer(ConvNeXtBlock, widths[1], widths[2], depths[1], True, dropout_p, dpr, cur)
        # Stage 3
        self.stage3, cur = self._make_layer(ConvNeXtBlock, widths[2], widths[3], depths[2], True, dropout_p, dpr, cur)
        # Stage 4
        self.stage4, cur = self._make_layer(ConvNeXtBlock, widths[3], widths[3], depths[3], False, dropout_p, dpr, cur)
        # Head 1x1xC4 Global Pool and classification
        self.norm = nn.LayerNorm(widths[-1], eps=1e-6)
        self.dropout = nn.Dropout(p=dropout_p) # Added Dropout before the linear layer
        # Modified the final linear layer to output 1 value for binary classification
        self.linear = nn.Linear(widths[-1], 1)

   def _make_layer(self, block, in_channels, out_channels, num_blocks, ds=True, dropout_p: float = 0.1, dpr: List[float] = None, cur: int = 0):
        layers = []
        for i in range(num_blocks):
            # Pass the next drop rate from the list to the block
            drop_rate = dpr[cur + i] if dpr else 0.0
            
            # Note: We pass the calculated 'drop_rate' to the 'drop_p' argument
            layers.append(block(in_channels, in_channels, drop_p=drop_rate, dropout_p=dropout_p))

        # Update the index for the next stage
        cur += num_blocks 
        
        if ds:
            layers.append(Downsample(in_channels, out_channels))

        return nn.Sequential(*layers), cur


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
        x = self.dropout(x) # Applied Dropout
        x = self.linear(x)
        return x