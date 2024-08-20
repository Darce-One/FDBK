import torch
from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import einops
from tqdm import tqdm

from fdbk_dataset_class import FDBK_Dataset
from network import Network


device = torch.device("mps")
print("Device: " + str(device))

DATAFRAME_PATH = 'fdbk_dataframe_normalised.csv'
BS = 30
EPOCHS = 10
LR = 1e-4
IN_FEATURES = 50
OUT_FEATURES = 8


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)


def train(dataset):
    model = Network(IN_FEATURES+1, OUT_FEATURES).to(device)
    model.apply(initialize_weights)
    model.train()
    train_loader = DataLoader(dataset, batch_size=BS, shuffle=True, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_func = nn.HuberLoss()


    for epoch in range(EPOCHS):
        for i, (x, y) in tqdm(enumerate(train_loader)):
            x, y = x.to(device), y.to(device)
            x = torch.cat((x, y[:, 0]), dim=1) # include original f0 in the feature vector
            optimizer.zero_grad()
            x = einops.rearrange(x, 'b f c -> b (c f)')
            y_pred = model(x)
            loss = loss_func(y_pred, y[:, 1:]) # remove original f0 from the target
            loss.backward()
            optimizer.step()
            if i % 50 == 0:
                print(f"Epoch: {epoch}, Loss: {loss.item()}")

    torch.save(model.state_dict(), 'trained_model.pth')
    return model


def test(model, dataset): # added the test func
    model.eval()
    test_loader = DataLoader(dataset, batch_size=BS, shuffle=True, drop_last=True)
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (x, y) in tqdm(enumerate(test_loader)):
            x, y = x.to(device), y[:, 1:].to(device)
            x = einops.rearrange(x, 'b f c -> b (c f)')
            y_pred = model(x)
            _, predicted = torch.max(y_pred, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    accuracy = correct / total
    return accuracy



def main():
    dataset = FDBK_Dataset(DATAFRAME_PATH)
    train_dataset, test_dataset = random_split(dataset, [0.9, 0.1])
    trained_model = train(train_dataset)
    accuracy = test(trained_model, test_dataset)
    print(f'Accuracy: {accuracy * 100:.2f}%')


if __name__ == "__main__":
    main()
