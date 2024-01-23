import torch.nn as nn
from DoubleConv import DoubleConv
import torch
import torchvision.transforms.functional as TF

class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(Unet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # contracting path
        prev_feature = in_channels
        for feature in features:
            self.downs.append(DoubleConv(prev_feature, feature))
            prev_feature = feature

        # expanding path
        for feature in list(reversed(features)):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2,
                    feature,
                    kernel_size=2,
                    stride=2
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        last_feature = features[-1]
        first_feature = features[0]
        self.bottom = DoubleConv(last_feature, last_feature*2)
        self.final = nn.Conv2d(first_feature, out_channels, kernel_size=1)

    def forward(self, x):
        connections = []
        for down in self.downs:
            x = down(x)
            connections.append(x)
            x = self.pool(x)

        x = self.bottom(x)
        # reverse connection
        connections = connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            connection = connections[idx//2]
            if x.shape != connection.shape:
                # consider making this crop instead of resize
                x = TF.resize(x, size=connection.shape[2:])
            # batch, channel, height, width?
            concat = torch.cat((connection, x), dim=1)
            x = self.ups[idx+1](concat)

        return self.final(x)

def test():
    x = torch.randn((3, 3, 160, 160))
    model = Unet(in_channels=3, out_channels=1)
    preds = model(x)
    print(x.shape)
    print(preds.shape)

if __name__ == "__main__":
    test()