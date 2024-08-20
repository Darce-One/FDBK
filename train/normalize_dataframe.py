import pandas as pd
import numpy as np


dataset = pd.read_csv('fdbk_dataframe.csv').to_numpy()[:, 2:]


dataset_length = len(dataset)

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

# Params: [add_amp, blip_amp, fm_amp, ratio, index, pnoise_amp, bnoise_amp, lpf_freq, hpf_freq]
mult = np.diag([1, 1, 1, 1/4.99, 1/2, 1, 1, 1/4950, 1/4950])
adds = np.array([0, 0, 0, -0.01, 0, 0, 0, -50, -50])
adds = np.tile(adds, (dataset_length, 1))

normalised = (dataset + adds) @ mult

#dataset = pd.DataFrame(normalised, columns=['add_amp', 'blip_amp', 'fm_amp', 'mfreq', 'index', 'pnoise_amp', 'bnoise_amp', 'lpf_freq', 'hpf_freq'])
dataset = pd.DataFrame(normalised, columns=['fund', 'amp_ratio', 'fm_ratio', 'fm_index', 'pnoise_amp', 'lpf_freq', 'osc_fm_ratio'])
dataset_full = pd.concat([pd.read_csv('fdbk_dataframe.csv').iloc[:, 0], dataset], axis=1)

print(dataset_full.head())

dataset_full.to_csv('fdbk_dataframe_normalised.csv', index=False)
