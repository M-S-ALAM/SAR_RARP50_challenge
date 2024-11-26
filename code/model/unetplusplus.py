import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)

class UNetPlusPlus(nn.Module):
    def __init__(self, n_channels=3, n_classes=10, features=[64, 128, 256, 512]):
        super(UNetPlusPlus, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.ups = nn.ModuleList()

        # Encoder
        in_channels = n_channels
        for feature in features:
            self.encoder.append(ConvBlock(in_channels, feature))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = feature

        # Decoder
        for i in range(len(features)):
            stage = nn.ModuleList()
            for j in range(i + 1):
                # Calculate the correct number of input channels for the ConvBlock
                if j == 0:
                    in_channels = features[i]
                else:
                    in_channels = features[i - j] + features[i - j + 1]

                if j < i:
                    stage.append(ConvBlock(in_channels, features[i - j - 1]))
                    self.ups.append(UpConv(features[i - j - 1], features[i - j - 1]))
                else:
                    stage.append(ConvBlock(in_channels, features[i - j]))

            self.decoder.insert(0, stage)  # insert at the beginning to match encoder order

        self.final_conv = nn.Conv2d(features[0], n_classes, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for i, (enc, pool) in enumerate(zip(self.encoder, self.pools)):
            x = enc(x)
            skip_connections.append(x)
            x = pool(x)

        for i, stage in enumerate(self.decoder):
            for j, conv in enumerate(stage):
                if j == 0:
                    x = self.ups[-(i + 1)](x)
                x = conv(x + skip_connections[-(i + j + 2)])  # concatenate skip connection and upsampled feature

        x = self.final_conv(x)
        return x

# Example
model = UNetPlusPlus(n_channels=3, n_classes=10)
input_tensor = torch.rand((1, 3, 256, 256))
output = model(input_tensor)
print(output.shape)  # Should be [1, 10, 256, 256]
