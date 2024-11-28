import torch
import torch.nn as nn
from torchvision.models import resnet50
from torch.nn import Transformer


class MATIS(nn.Module):
    def __init__(self, num_classes=10, embed_dim=256, num_heads=8, num_encoder_layers=6):
        super(MATIS, self).__init__()

        # Backbone (Feature Extractor)
        backbone = resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # Remove FC layers

        # Projection layer to align feature dimensions with embedding dimension
        self.projection = nn.Conv2d(2048, embed_dim, kernel_size=1)  # Align channels to embed_dim

        # Positional Encoding
        self.position_encoding = nn.Parameter(torch.randn(1, embed_dim, 50, 50))  # Assume 50x50 feature map

        # Transformer
        self.transformer = Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=512
        )

        # Decoder
        self.decoder = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, x):
        # Extract features
        features = self.backbone(x)  # Shape: [B, 2048, H, W]

        # Apply projection to match positional encoding dimension
        features = self.projection(features)  # Shape: [B, 256, H, W]

        # Add positional encoding
        b, c, h, w = features.size()
        features = features + self.position_encoding[:, :, :h, :w].to(features.device)

        # Reshape for Transformer
        features = features.flatten(2).permute(2, 0, 1)  # Shape: [H*W, B, C]

        # Apply Transformer
        transformed_features = self.transformer(features, features)  # Shape: [H*W, B, C]

        # Reshape back to spatial dimensions
        transformed_features = transformed_features.permute(1, 2, 0).view(b, c, h, w)

        # Decode to segmentation map
        segmentation_map = self.decoder(transformed_features)  # Shape: [B, num_classes, H, W]
        return segmentation_map


if __name__=='__main__':
    model = MATIS()
    print(model)