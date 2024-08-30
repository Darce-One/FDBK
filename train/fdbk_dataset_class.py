import torch
import librosa
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import essentia
import essentia.standard as es

NUM_MFCCS = 40

class FDBK_Dataset(Dataset):
    def __init__(self, csv, sample_rate, block_size, mode):
        self.dataframe = pd.read_csv(csv)
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.mode = mode


        # Essentia algorithm instances defined here to save memory
        self.w = es.Windowing(type='hann', size=self.block_size)
        self.spectrum = es.Spectrum(size=self.block_size)
        self.spectral_peaks = es.SpectralPeaks(maxPeaks=10, minFrequency=30, sampleRate=self.sample_rate) #https://essentia.upf.edu/reference/std_SpectralPeaks.html
        self.pitch_detect = es.PitchYinFFT(sampleRate=self.sample_rate, frameSize=self.block_size)
        self.mfcc = es.MFCC(numberCoefficients=NUM_MFCCS, inputSize=int(self.block_size/2)+1)
        self.spectral_contrast = es.SpectralContrast(sampleRate=self.sample_rate, frameSize=self.block_size)
        self.inharmonicity = es.Inharmonicity()
        self.dissonance = es.Dissonance()
        self.pitch_salience = es.PitchSalience(sampleRate=self.sample_rate)
        self.flatness = es.Flatness()


    def _load_audio(self, path):
        y, sr = librosa.load(path, sr=self.sample_rate, dtype=np.float32)
        return y, sr

    def _extract_features(self, y):
        features = np.array([])
        spectrum = self.spectrum(self.w(y))
        if self.mode == 'mfcc':
            _, features = self.mfcc(spectrum)
            return features
        elif self.mode == 'feature':
            freqs, mags = self.spectral_peaks(spectrum)
            melbands, mfcc = self.mfcc(spectrum)
            spectral_contrast, _ = self.spectral_contrast(spectrum)
            inharmonicity = self.inharmonicity(freqs, mags)
            dissonance = self.dissonance(freqs, mags)
            pitch_salience = self.pitch_salience(spectrum)
            flatness = self.flatness(spectrum)
            return np.concatenate((mfcc, melbands, spectral_contrast, [inharmonicity, dissonance, pitch_salience, flatness]), dtype=np.float32)
        else:
            raise ValueError(f"mode '{self.mode}' not recognised, please use one of the following: ['mfcc', 'feature']")
        return features

    def _resize_if_necessary(self, audio, shape):
        if audio.size < shape:
            print("found short sample in dataset, only a problem if frequent")
            zeros = np.zeros((shape), dtype=np.float32)
            zeros[:audio.size] += audio
            audio = zeros
        else:
            audio = audio[:shape]
        return audio

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        audio, sr = self._load_audio(self.dataframe.iloc[idx, 0])
        audio = self._resize_if_necessary(audio, self.block_size)
        features = np.array(self._extract_features(audio), dtype=np.float32)
        synth_parameters = np.array(self.dataframe.iloc[idx].values[1:], dtype=np.float32)

        return features, synth_parameters
