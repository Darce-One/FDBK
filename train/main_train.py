import argparse
from pandas.core.indexing import check_dict_or_set_indexers
import torch
from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import einops
from tqdm import tqdm
import os
import json
import pandas as pd
import numpy as np

from fdbk_dataset_class import FDBK_Dataset
from network import Network

device = torch.device("mps")
print("Device: " + str(device))


SAMPLES_PATH = './dataset/samples'
JSON_FILE_PATH = './dataset/params.json'
CSV_FILE_PATH = './dataset/fdbk_dataframe.csv'
NORM_CSV_FILE_PATH = './dataset/fdbk_dataframe_normalised.csv'

BS = 30
EPOCHS = 2
LR = 1e-4
IN_FEATURES = 50
OUT_FEATURES = 7


def create_dataframe(folder_path, json_file_path, csv_file_path_save):
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    with open(json_file_path, 'r') as json_file:
        json_data = json.load(json_file)
        parameters = json_data.get('data', {})
        # print(parameters)

    data = []
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        relative_name = f"sound-{filename.split('.')[0].split('-')[-1]}"
        param = parameters.get(relative_name, None)
        # print(param)
        if param is not None:
            entry = {'file_path': file_path}
            for i, p in enumerate(param):
                entry[f'parameter_{i+1}'] = p
            data.append(entry)

    df = pd.DataFrame(data)
    #print(df.head())
    df.to_csv(csv_file_path_save, index=False)


def normalize_dataframe(csv_file_path, norm_csv_file_path):
    dataset = pd.read_csv(csv_file_path).to_numpy()[:, 1:]
    dataset_length = len(dataset)
    #print(dataset.head())
    """
    fund = exprand(50.0, 2000.0);
    amp_ratio = rrand(0.5, 10.0);
    fm_ratio = rrand(0.25, 5.0);
    fm_index = rrand(0.0, 2.0);
    pnoise_amp = rrand(0.0, 1.0);
    lpf_freq = exprand(50, 5000.0);
    hpf_freq = exprand(50.0, 5000);
    osc_fm_ratio = rrand(0.05, 0.95);
    """
    mult = np.diag([1/1950, 1/9.5, 1/4.75, 1/2, 1, 1/4950, 1/4950, 1/0.9])
    adds = np.array([-50, -0.5, 0.25, 0, 0, -50, -50, -0.05])
    adds = np.tile(adds, (dataset_length, 1))
    normalised = (dataset + adds) @ mult

    dataset = pd.DataFrame(normalised, columns=['fund', 'amp_ratio', 'fm_ratio', 'fm_index', 'pnoise_amp', 'lpf_freq', 'hpf_freq', 'osc_fm_ratio'])
    dataset_full = pd.concat([pd.read_csv(csv_file_path).iloc[:, 0], dataset], axis=1)
    print(dataset_full.head())
    dataset_full.to_csv(norm_csv_file_path, index=False)


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)


def train(dataset, mode):
    in_features = 0
    out_features = OUT_FEATURES
    if mode == 'mfcc':
        in_features = 40
    elif mode == 'feature':
        in_features = 50
    model = Network(in_features+1, out_features).to(device)
    model.apply(initialize_weights)
    model.train()
    train_loader = DataLoader(dataset, batch_size=BS, shuffle=True, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_func = nn.HuberLoss()


    for epoch in range(EPOCHS):
        for i, (x, y) in tqdm(enumerate(train_loader)):
            x, y = x.to(device), y.to(device)
            x = torch.cat((x, y[:, 0].unsqueeze(1)), dim=1) # include original f0 in the feature vector
            optimizer.zero_grad()
            # x = einops.rearrange(x, 'b f c -> b (c f)')
            y_pred = model(x)
            loss = loss_func(y_pred, y[:, 1:]) # remove original f0 from the target
            loss.backward()
            optimizer.step()
            if i % 50 == 0:
                print(f"Epoch: {epoch}, Loss: {loss.item()}")

    torch.save(model.state_dict(), f"dataset/trained_model_{mode}.pth")
    return model


def test(model, dataset):
    model.eval()
    loss_func = nn.HuberLoss()
    test_loader = DataLoader(dataset, batch_size=BS, shuffle=True, drop_last=True)
    cum_loss = 0
    avg_loss = 0

    with torch.no_grad():
        for i, (x, y) in tqdm(enumerate(test_loader)):
            x, y = x.to(device), y.to(device)
            x = torch.cat((x, y[:, 0].unsqueeze(1)), dim=1) # include original f0 in the feature vector
            y_pred = model(x)
            loss = loss_func(y_pred, y[:, 1:])
            cum_loss += loss
            avg_loss = cum_loss / i

    return avg_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sample_rate', type=int, default=44100, help="sets the project sample rate")
    parser.add_argument('-b', '--block_size', type=int, default=2048, help="sets the project block_size. Make sure to use a multiple of 2")
    parser.add_argument('-m', '--mode', choices=['mfcc', 'feature'], help="choose between the 'mfcc' and 'feature' mode")
    args = parser.parse_args()
    assert args.mode is not None, "Please choose a mode using '-m' or '--mode', with one of the following options: ['mfcc', 'feature']"

    create_dataframe(os.path.abspath(SAMPLES_PATH), os.path.abspath(JSON_FILE_PATH), os.path.abspath(CSV_FILE_PATH))
    normalize_dataframe(CSV_FILE_PATH, NORM_CSV_FILE_PATH)

    dataset = FDBK_Dataset(NORM_CSV_FILE_PATH, args.sample_rate, args.block_size, mode=args.mode)
    train_dataset, test_dataset = random_split(dataset, [0.9, 0.1])
    trained_model = train(train_dataset, mode=args.mode)
    avg_loss = test(trained_model, test_dataset)
    print(f'average loss: {avg_loss}')
    print("Note that this loss is to be interpreted in context with other training runs. It is not an objective metric")


if __name__ == "__main__":
    main()
