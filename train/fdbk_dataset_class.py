import torch
import librosa
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

DATAFRAME_PATH = 'fdbk_dataframe_final_normalised.csv'
NUM_MFCCS = 40
SAMPLES = 8192

class FDBK_Dataset(Dataset):
    def __init__(self, csv):
        self.dataframe = pd.read_csv(csv)


    def _load_audio(self, path):
        y, sr = librosa.load(path, sr=None)
        assert type(y) == np.ndarray, "y is not a numpy array"
        return y, sr

    def _extract_features(self, y, sr):
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=NUM_MFCCS, win_length=8192, hop_length=8194, n_fft=8192)
        return mfccs

    def _resize_if_necessary(self, audio, shape):
        if audio.size < shape:
            zeros = np.zeros((shape))
            zeros[:audio.size] += audio
            # audio = np.pad(audio, ((0, 0), (shape - audio.size)), mode='constant')
            audio = zeros
        else:
            audio = audio[:shape]
        return audio

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        audio, sr = self._load_audio(self.dataframe.iloc[idx, 0])
        audio = self._resize_if_necessary(audio, SAMPLES)
        mfccs = np.array(self._extract_features(audio, sr), dtype=np.float32)
        synth_parameters = np.array(self.dataframe.iloc[idx].values[1:], dtype=np.float32)

        return mfccs, synth_parameters


if __name__ == '__main__':
    ad = FDBK_Dataset(DATAFRAME_PATH)
    print(ad[0][0].shape)
