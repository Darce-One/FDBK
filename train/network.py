import torch
import librosa
import time
import torch.nn as nn


device = torch.device("mps")
in_features = 50
out_features = 8


class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout, activation):
        super(LinearBlock, self).__init__()

        activation_functions = {
                'relu': nn.ReLU(inplace=True),
                'leaky_relu': nn.LeakyReLU(inplace=True),
                'tanh': nn.Tanh(),
                'none': nn.Identity()
                }

        layers = [
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            activation_functions[activation]
        ]

        if dropout > 0:
            layers.insert(2, nn.Dropout(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        return x


class Network(nn.Module):
    def __init__(self, in_features, out_features):
        super(Network, self).__init__()

        self.layer1 = LinearBlock(in_features, 140, 0.1, 'leaky_relu')
        self.layer2 = LinearBlock(140, 90, 0.1, 'leaky_relu')
        self.layer3 = LinearBlock(90, 50, 0.1, 'leaky_relu')
        self.layer4 = LinearBlock(50, 30, 0, 'tanh')
        self.layer5 = LinearBlock(30, out_features, 0, 'none')
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.sigmoid(x)
        return x



if __name__ == "__main__":
    model = Network(in_features, out_features).to(device)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
