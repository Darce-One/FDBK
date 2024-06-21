import torch
import librosa
import time
import torch.nn as nn


N_MFCCs = 40

device = torch.device("mps")


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
    def __init__(self):
        super(Network, self).__init__()

        self.layer1 = LinearBlock(N_MFCCs, 100, 0.2, 'leaky_relu')
        self.layer2 = LinearBlock(100, 30, 0.2, 'leaky_relu')
        self.layer3 = LinearBlock(30, 9, 0, 'none')

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.sigmoid(x)
        return x




if __name__ == "__main__":
    model = Network().to(device)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
