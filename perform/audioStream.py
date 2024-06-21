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
from network import Network


SAMPLE_RATE = 44100
BLOCK_SIZE = 2048 * 4
CHANNELS = 2
MODEL_PATH = 'final_trained_model.pth'

class AudioProcessor():
    def __init__(self, sample_rate: int, n_mfcc: int = 40) -> None:
        self.sample_rate: int = sample_rate
        self.n_mfcc = n_mfcc
        self.client = udp_client.SimpleUDPClient("127.0.0.1", 57120)
        self.model = Network()
        self.model.load_state_dict(torch.load(MODEL_PATH))
        self.model.eval()
        self.w = es.Windowing(type='hann')
        self.spectrum = es.Spectrum()
        self.pitch_detect = es.PitchYinFFT()

    def prepare_to_play(self) -> None:
        pass

    def process_block(self, buffer: np.ndarray) -> None:
        if len(buffer) == 2*BLOCK_SIZE:
            audio = np.vstack((buffer[::2], buffer[1::2]))
        else:
            audio = np.transpose(buffer[:, np.newaxis])

        channel_params = np.zeros((audio.shape[0], 10))

        for iter, channel in enumerate(audio):
            mfcc = librosa.feature.mfcc(y=channel, sr=self.sample_rate, n_mfcc=self.n_mfcc, win_length=BLOCK_SIZE, hop_length=BLOCK_SIZE+2, n_fft=BLOCK_SIZE)

            buf_spec = self.spectrum(self.w(channel))
            f0 = self.pitch_detect(buf_spec)[0]
            params = self.model(torch.tensor(mfcc.reshape(1, 40)))

            channel_params[(iter + 1) % 2, 0] = float(f0)
            channel_params[iter, 1] = float(params[0][0])
            channel_params[iter, 2] = float(params[0][1])
            channel_params[iter, 3] = float(params[0][2])
            channel_params[iter, 4] = float(params[0][3] * 4.99 + 0.01)
            channel_params[iter, 5] = float(params[0][4] * 2)
            channel_params[iter, 6] = float(params[0][5])
            channel_params[iter, 7] = float(params[0][6])
            channel_params[iter, 8] = float(params[0][7] * 4950 + 50)
            channel_params[iter, 9] = float(params[0][8] * 4950 + 50)

        for iter in range(CHANNELS):
            address = f"/channel_{iter}"  # OSC address pattern
            msg = osc_message_builder.OscMessageBuilder(address=address)

            for par in channel_params[iter]:
                msg.add_arg(par)

            msg = msg.build()  # Build the message

            # Step 3: Send the OSC message
            self.client.send(msg)



        # mfcc = librosa.feature.mfcc(y=buffer, sr=self.sample_rate, n_mfcc=self.n_mfcc, win_length=8192, hop_length=8194, n_fft=8192)
        # get F0
        # buf_spec = self.spectrum(self.w(buffer))
        # f0 = self.pitch_detect(buf_spec)[0]

        # Run through model
        # params = self.model(torch.tensor(mfcc.reshape(1, 40)))
        # address = "/test"  # OSC address pattern
        # msg = osc_message_builder.OscMessageBuilder(address=address)
        # msg.add_arg(float(params[0][0] * 2950 + 50))  # Add an argument to the message
        # msg.add_arg(float(f0))
        # msg.add_arg(float(params[0][1]))
        # msg.add_arg(float(params[0][2]))
        # msg.add_arg(float(params[0][3] * 9 + 1))
        # msg.add_arg(float(params[0][4] * 2 + 1))
        # msg.add_arg(float(params[0][5]))
        # msg.add_arg(float(params[0][6] * 4900 + 100))

        # msg = msg.build()  # Build the message

        # Step 3: Send the OSC message
        # self.client.send(msg)



def main():
    # instanciate the Audio Processor
    audio_processor = AudioProcessor(SAMPLE_RATE)

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
