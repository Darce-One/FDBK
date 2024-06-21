import pandas as pd
import numpy as np


dataset = pd.read_csv('fdbk_dataframe_final.csv').to_numpy()[:, 2:]


dataset_length = len(dataset)

# Params: [add_amp, blip_amp, fm_amp, ratio, index, pnoise_amp, bnoise_amp, lpf_freq, hpf_freq]
mult = np.diag([1, 1, 1, 1/4.99, 1/2, 1, 1, 1/4950, 1/4950])
adds = np.array([0, 0, 0, -0.01, 0, 0, 0, -50, -50])
adds = np.tile(adds, (dataset_length, 1))

normalised = (dataset + adds) @ mult

dataset = pd.DataFrame(normalised, columns=['add_amp', 'blip_amp', 'fm_amp', 'mfreq', 'index', 'pnoise_amp', 'bnoise_amp', 'lpf_freq', 'hpf_freq'])
dataset_full = pd.concat([pd.read_csv('fdbk_dataframe_final.csv').iloc[:, 0], dataset], axis=1)

print(dataset_full.head())

dataset_full.to_csv('fdbk_dataframe_final_normalised.csv', index=False)

'''
var fund = exprand(50.0, 3000.0);
		var add_amp = rrand(0.0, 1.0);
		var fm_amp = rrand(0.0, 1.0);
		var mfreq = exprand(1.0, 10);
		var index = rrand(1.0, 3.0);
		var noise_amp = rrand(0.0, 1.0);
		var filt_freq = exprand(100.0, 5000.0);'''
