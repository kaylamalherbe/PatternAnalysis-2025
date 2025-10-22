from torchvision.ops import StochasticDepth
import torch
from torch import nn, Tensor
from typing import List

class ConvNeXtBlock(nn.Module):
    def __init__(self, in_channels: int, drop_p: float = .0):
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

def downsample_layer(in_channels: int,
                     out_channels: int) -> nn.Module: # Changed to nn.Module for better clarity, though Sequential works
    
    # LayerNorm needs to be applied over the channel dimension (last dimension)
    norm = nn.LayerNorm(in_channels, eps=1e-6, elementwise_affine=True)
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    class Downsample(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            # 1. Permute to (B, H, W, C) for LayerNorm
            x = x.permute(0, 2, 3, 1) 
            # 2. Apply LayerNorm over channels
            x = norm(x)
            # 3. Permute back to (B, C, H, W) for Conv2d
            x = x.permute(0, 3, 1, 2)
            # 4. Apply Conv2d
            x = conv(x)
            return x
            
    return Downsample()

def global_avg_pool(x: Tensor) -> Tensor:
    return x.mean(dim=(-2, -1), keepdim=True)   


def linear_layer(in_features: int,
                 out_features: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(in_features, eps=1e-6, elementwise_affine=True),
        nn.Linear(in_features, out_features)
    )

def ConvNextEncoder(in_channels: int,
                    stem_features: int,
                    depths: List[int],
                    widths: List[int],
                    drop_p: float = .0) -> nn.Sequential:
    
    layers = []
    num_stages = len(depths)
    total_blocks = sum(depths)
    block_id = 0

    stem_conv = nn.Conv2d(in_channels, stem_features, kernel_size=4, stride=4)
    stem_norm = nn.LayerNorm(stem_features, eps=1e-6, elementwise_affine=True)
    
    class Stem(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            # 1. Conv2d
            x = stem_conv(x)
            # 2. Permute for LayerNorm (B, C, H, W) -> (B, H, W, C)
            x = x.permute(0, 2, 3, 1)
            # 3. LayerNorm
            x = stem_norm(x)
            # 4. Permute back (B, H, W, C) -> (B, C, H, W)
            x = x.permute(0, 3, 1, 2)
            return x
    
    layers.append(Stem())

    # Stages
    for stage in range(num_stages):
        stage_layers = []
        if stage > 0:
            stage_layers.append(downsample_layer(widths[stage - 1], widths[stage]))
        
        for _ in range(depths[stage]):
            drop_prob = drop_p * block_id / total_blocks
            # NOTE: widths[stage] is the channel count for the block
            stage_layers.append(ConvNeXtBlock(widths[stage], drop_prob)) 
            block_id += 1
        
        layers.append(nn.Sequential(*stage_layers))

    
    return nn.Sequential(*layers)


# use covnext encoder to predict
class ClassificationHead(nn.Module):
    def __init__(self, num_channels: int, num_classes: int = 1000):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=1e-6)
        self.linear = nn.Linear(num_channels, num_classes)
        
    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(1) 
        x = self.norm(x)
        x = self.linear(x)
        return x

# prediction    
class ConvNextForImageClassification(nn.Module): # Use nn.Module instead of nn.Sequential
    def __init__(self, 
                 in_channels: int,
                 stem_features: int,
                 depths: List[int],
                 widths: List[int],
                 drop_p: float = .0,
                 num_classes: int = 1000):
        super().__init__()
        self.encoder = ConvNextEncoder(in_channels, stem_features, depths, widths, drop_p)
        self.head = ClassificationHead(widths[-1], num_classes)
        
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x) # Output is (B, C, H, W) e.g., (B, 768, 7, 7)
        x = global_avg_pool(x) 
        x = self.head(x) # Output is (B, num_classes)
        x = torch.softmax(x, dim=1) 
        
        return x
    
# def main
if __name__ == "__main__":
    model = ConvNextForImageClassification(
        in_channels=3,
        stem_features=96,
        depths=[3, 3, 9, 3],
        widths=[96, 192, 384, 768],
        drop_p=0.1,
        num_classes=1000
    )
    
    sample_input = torch.randn(1, 3, 224, 224)  # Example input tensor
    output = model(sample_input)
    print(output.shape)  # Should print torch.Size([1, 1000])

model = ConvNextForImageClassification(
      in_channels=3,
      stem_features=96,
      depths=[3, 3, 9, 3], 
      widths=[96, 192, 384, 768],
      drop_p=0.1,
      num_classes=2 # Changed to 2 for binary classification (AD/NC)
  )
#e test
# # Get a batch of images from the dataloader
# images, labels = next(iter(dl))

# # Pass the batch of images to the model
# output = model(images)
# print(output.shape)