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
        self.w = es.Windowing(type='hann')
        self.spectrum = es.Spectrum(size=self.block_size)
        self.spectral_peaks = es.SpectralPeaks(maxPeaks=10, minFrequency=30, sampleRate=self.sample_rate) #https://essentia.upf.edu/reference/std_SpectralPeaks.html
        self.pitch_detect = es.PitchYinFFT()
        self.mfcc = es.MFCC(numberCoefficients=NUM_MFCCS)
        self.melbands = es.MelBands(numberBands=40, lowFrequencyBound=50, highFrequencyBound=8000)
        self.spectral_contrast = es.SpectralContrast()
        self.inharmonicity = es.Inharmonicity()
        self.dissonance = es.Dissonance()
        self.pitch_salience = es.PitchSalience()
        self.flatness = es.Flatness()


    def _load_audio(self, path):
        y, sr = librosa.load(path, sr=self.sample_rate)
        return y, sr

    def _extract_features(self, y):
        features = np.array([])
        spectrum = self.spectrum(self.w(y))
        if self.mode == 'mfcc':
            _, features = self.mfcc(spectrum)
            return features
        elif self.mode == 'feature':
            freqs, mags = self.spectral_peaks(spectrum)
            _, mfcc = self.mfcc(spectrum)
            melbands_spectrum = self.melbands(spectrum)
            spectral_contrast, _ = self.spectral_contrast(spectrum)
            inharmonicity = self.inharmonicity(freqs, mags)
            dissonance = self.dissonance(freqs, mags)
            pitch_salience = self.pitch_salience(spectrum)
            flatness = self.flatness(spectrum)
            return np.concatenate((mfcc, melbands_spectrum, spectral_contrast, [inharmonicity, dissonance, pitch_salience, flatness]), dtype=np.float32)
        else:
            raise ValueError(f"mode '{self.mode}' not recognised, please use one of the following: ['mfcc', 'feature']")
        return features

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
        audio = self._resize_if_necessary(audio, self.block_size)
        features = np.array(self._extract_features(audio), dtype=np.float32)
        synth_parameters = np.array(self.dataframe.iloc[idx].values[1:], dtype=np.float32)
        #print(f"features: {features.shape}, synth_parameters: {synth_parameters.shape}")

        return features, synth_parameters


