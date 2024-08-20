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
from ..train.network import Network
import argparse


SAMPLE_RATE = 44100
BLOCK_SIZE = 2048 * 1
CHANNELS = 1
MODEL_PATH = 'trained_model.pth'
# Mode possibilities: 'mfcc', 'feature'
MODE = 'mfcc'

class AudioProcessor():
    def __init__(self, sample_rate: int, block_size:int, num_channels:int, mode:str) -> None:
        self.sample_rate: int = sample_rate
        self.block_size = block_size
        self.num_channels = num_channels
        self.mode = mode
        self.n_mfcc = 40
        self.client = udp_client.SimpleUDPClient("127.0.0.1", 57120)
        self.model = Network()
        self.model.load_state_dict(torch.load(MODEL_PATH))
        self.model.eval()
        self.w = es.Windowing(type='hann')
        self.spectrum = es.Spectrum(size=self.block_size)
        self.spectral_peaks = es.SpectralPeaks(maxPeaks=10, minFrequency=30, sampleRate=self.sample_rate) #https://essentia.upf.edu/reference/std_SpectralPeaks.html
        self.pitch_detect = es.PitchYinFFT()

    def extract_features(self, audio) -> np.ndarray:
        features = np.array([])
        if self.mode == 'mfcc':
            features = librosa.feature.mfcc(y=audio, sr=self.sample_rate,
                n_mfcc=self.n_mfcc, win_length=self.block_size,
                hop_length=self.block_size+2, n_fft=self.block_size)
        elif self.mode == 'feature':
            # TODO: Extract features here and fewer MFCCs.
            #
            # https://essentia.upf.edu/reference/std_LPC.html
            # FlatnessDB, HFC,
            spectrum = self.spectrum(self.w(audio))
            freqs, mags = self.spectral_peaks(spectrum)
            mfcc_bands, mfcc_coeffs = es.array(es.MFCC(spectrum)).T
            spectral_contrast, _ = es.SpectralContrast(spectrum)
            inharmonicity = es.Inharmonicity(freqs, mags)
            dissonance = es.Dissonance(freqs, mags)
            pitch_salience = es.PitchSalience(spectrum)
            flatness = es.Flatness(spectrum)
            raise NotImplementedError
        return features



    def process_block(self, buffer: np.ndarray) -> None:
        if len(buffer) == 2*self.block_size:
            audio = np.vstack((buffer[::2], buffer[1::2]))
        else:
            audio = np.transpose(buffer[:, np.newaxis])

        channel_params = np.zeros((audio.shape[0], 10))

        for iter, channel in enumerate(audio):
            features = self.extract_features(channel)

            buf_spec = self.spectrum(self.w(channel))
            f0 = self.pitch_detect(buf_spec)[0]
            params = self.model(torch.tensor(features.reshape(1, 40)))

            channel_params[(iter + 1) % self.num_channels, 0] = float(f0)
            channel_params[iter, 1] = float(params[0][0])
            channel_params[iter, 2] = float(params[0][1])
            channel_params[iter, 3] = float(params[0][2])
            channel_params[iter, 4] = float(params[0][3] * 4.99 + 0.01)
            channel_params[iter, 5] = float(params[0][4] * 2)
            channel_params[iter, 6] = float(params[0][5])
            channel_params[iter, 7] = float(params[0][6])
            channel_params[iter, 8] = float(params[0][7] * 4950 + 50)
            channel_params[iter, 9] = float(params[0][8] * 4950 + 50)

        for iter in range(self.num_channels):
            address = f"/channel_{iter}"  # OSC address pattern
            msg = osc_message_builder.OscMessageBuilder(address=address)

            for par in channel_params[iter]:
                msg.add_arg(par)

            msg = msg.build()  # Build the message

            # Step 3: Send the OSC message
            self.client.send(msg)

def map_range(value, old_min=0, old_max=1, new_min=0, new_max=1):
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

    # instanciate the Audio Processor
    audio_processor = AudioProcessor(args.sample_rate, args.block_size, args.channel_count, args.mode)

    # Function to process audio data in real-time
    def process_audio(in_data, frame_count, time_info, status):
        if in_data is None:
                print("Received None for in_data")
        # Convert the input data to a numpy array
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        # out = np.zeros_like(audio_data)[:, np.newaxis]
        # out = np.zeros((frame_count, 1), dtype=np.float32)
        # print(out.shape)

        audio_processor.process_block(audio_data)


        # Convert the processed data back to bytes
        out_data = np.zeros_like(audio_data)[:, np.newaxis].tobytes()

        # out_data = audio_data.tobytes()

        return out_data, pyaudio.paContinue

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open a stream with the specified callback function
    stream = p.open(format=pyaudio.paFloat32,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    frames_per_buffer=BLOCK_SIZE,
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
