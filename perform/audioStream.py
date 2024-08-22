import pyaudio
import time
import numpy as np
from pythonosc import udp_client
from pythonosc import osc_message_builder
import librosa
import einops
import essentia
import essentia.standard as es
import torch
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.network import Network
import argparse

class AudioProcessor():
    def __init__(self, sample_rate: int, block_size:int, num_channels:int, mode:str) -> None:
        self.sample_rate: int = sample_rate
        self.block_size = block_size
        self.num_channels = num_channels
        self.mode = mode
        self.n_mfcc = 40
        self.client = udp_client.SimpleUDPClient("127.0.0.1", 57120)
        if self.mode == 'mfcc':
            self.in_features = 41
        else:
            self.in_features = 51
        self.out_features = 7
        self.model = Network(self.in_features, self.out_features)
        self.model.load_state_dict(torch.load(f"dataset/trained_model_{mode}.pth"))
        self.model.eval()

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

    def extract_features(self, audio) -> np.ndarray:
        features = np.array([])
        spectrum = self.spectrum(self.w(audio))
        f0 = self.pitch_detect(spectrum)[0]
        if self.mode == 'mfcc':
            """
            features = librosa.feature.mfcc(y=audio, sr=self.sample_rate,
                n_mfcc=self.n_mfcc, win_length=self.block_size,
                hop_length=self.block_size+2, n_fft=self.block_size)
            """
            _, features = self.mfcc(spectrum)
        elif self.mode == 'feature':
            freqs, mags = self.spectral_peaks(spectrum)

            _, mfcc = self.mfcc(spectrum)
            spectral_contrast, _ = self.spectral_contrast(spectrum)
            inharmonicity = self.inharmonicity(freqs, mags)
            dissonance = self.dissonance(freqs, mags)
            pitch_salience = self.pitch_salience(spectrum)
            flatness = self.flatness(spectrum)
            # concat everything into a single array
            features = np.concatenate((mfcc, spectral_contrast, [inharmonicity, dissonance, pitch_salience, flatness]), dtype=np.float32)

        # Appending the fundamental frequency to the features.
        features = np.concatenate((features, [f0]), dtype=np.float32)
        return features

    def get_usable_buffer(self, in_data) -> np.ndarray:
        in_data = np.frombuffer(in_data, dtype=np.float32)
        usable_buffer = np.zeros((self.num_channels, self.block_size), dtype=np.float32)
        for channel in range(self.num_channels):
            usable_buffer[channel, :] = in_data[channel::self.num_channels]
        return usable_buffer

    def process_block(self, buffer: np.ndarray) -> None:
        # initialise channel-wise parameters
        channel_params = np.zeros((self.num_channels, 8))

        for iter, channel in enumerate(buffer):
            features = self.extract_features(channel)
            params = self.model(torch.tensor(features).unsqueeze(0))[0]

            channel_params[(iter + 1) % self.num_channels, 0] = float(features[-1])
            channel_params[iter, 1] = float(map_range(params[0], new_min=0.5, new_max=10.0))  # amp_ratio = rrand(0.5, 10.0);
            channel_params[iter, 2] = float(map_range(params[1], new_min=0.25, new_max=5.0))  # fm_ratio = rrand(0.25, 5.0);
            channel_params[iter, 3] = float(map_range(params[2], new_min=0.0, new_max=2.0))   # fm_index = rrand(0.0, 2.0);
            channel_params[iter, 4] = float(map_range(params[3], new_min=0.0, new_max=1.0))   # pnoise_amp = rrand(0.0, 1.0);
            channel_params[iter, 5] = float(map_range(params[4], new_min=50., new_max=5000.)) # lpf_freq = exprand(50, 5000.0);
            channel_params[iter, 6] = float(map_range(params[5], new_min=50., new_max=5000.)) # hpf_freq = exprand(50.0, 5000.0);
            channel_params[iter, 7] = float(map_range(params[6], new_min=0.05, new_max=0.95)) # osc_fm_ratio = rrand(0.05, 0.95);


        for iter in range(self.num_channels):
            address = f"/channel_{iter}"  # OSC address pattern
            msg = osc_message_builder.OscMessageBuilder(address=address)

            for par in channel_params[iter]:
                msg.add_arg(par)

            msg = msg.build()  # Build the message

            # Step 3: Send the OSC message
            self.client.send(msg)

def map_range(value, old_min=0., old_max=1., new_min=0., new_max=1.):
    # This function linearly maps a value from one range to another
    norm_val = (value - old_min) / (old_max - old_min)
    scaled_arr = norm_val * (new_max - new_min) + new_min
    return scaled_arr



def main():
    # Define arguments
    parser = argparse.ArgumentParser(description="FDBK")
    parser.add_argument('-s', '--sample_rate', type=int, default=44100, help="sets the project sample rate")
    parser.add_argument('-b', '--block_size', type=int, default=2048, help="sets the project block_size. Make sure to use a multiple of 2")
    parser.add_argument('-c', '--channel_count', type=int, default=2, help="number of channels")
    parser.add_argument('-m', '--mode', choices=['mfcc', 'feature'], help="choose between the 'mfcc' and 'feature' mode")

    # Parse args
    args = parser.parse_args()

    assert args.mode is not None, "Please choose a mode using '-m' or '--mode', with one of the following options: ['mfcc', 'feature']"

    # instanciate the Audio Processor
    audio_processor = AudioProcessor(args.sample_rate, args.block_size, args.channel_count, args.mode)

    # Function to process audio data in real-time
    def process_audio(in_data, frame_count, time_info, status):
        if in_data is None:
                print("Received None for in_data")

        usable_buffer = audio_processor.get_usable_buffer(in_data)

        audio_processor.process_block(usable_buffer)


        # Convert the processed data back to bytes
        out_data = np.zeros_like(usable_buffer.T).tobytes()

        return out_data, pyaudio.paContinue

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open a stream with the specified callback function
    stream = p.open(format=pyaudio.paFloat32,
                    channels=args.channel_count,
                    rate=args.sample_rate,
                    frames_per_buffer=args.block_size,
                    input=True,
                    output=False,
                    stream_callback=process_audio)


    print("Stream open!")
    while stream.is_active():
        time.sleep(1)

    # Stop and close the stream
    stream.close()

    # Terminate PyAudio
    p.terminate()





if __name__ == "__main__":
    main()
