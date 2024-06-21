from fdbk_dataset_class import FDBK_Dataset
from network import Network
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import einops
from tqdm import tqdm

device = torch.device("mps")
print("Device: " + str(device))

DATAFRAME_PATH = 'fdbk_dataframe_normalised.csv'
BS = 30
EPOCHS = 10
LR = 1e-4

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)


def train():
    model = Network().to(device)
    model.apply(initialize_weights)
    dataloader = FDBK_Dataset(DATAFRAME_PATH)
    train_loader = DataLoader(dataloader, batch_size=BS, shuffle=True, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_func = nn.MSELoss()

    for epoch in range(EPOCHS):
        for i, (x, y) in tqdm(enumerate(train_loader)):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            x = einops.rearrange(x, 'b f c -> b (c f)')
            y_pred = model(x)
            loss = loss_func(y_pred, y)
            loss.backward()
            optimizer.step()
            if i % 50 == 0:
                print(f"Epoch: {epoch}, Loss: {loss.item()}")

    torch.save(model.state_dict(), 'trained_model.pth')



if __name__ == "__main__":
    train()
