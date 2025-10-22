
class ClassificationHead(nn.Sequential):
    def __init__(self, num_channels: int, num_classes: int = 1000):
        super().__init__(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.LayerNorm(num_channels),
            nn.Linear(num_channels, num_classes)
        )
    
    
class ConvNextForImageClassification(nn.Sequential):
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