import librosa
import torch
from basic_pitch.inference import ICASSP_2022_MODEL_PATH, Model, window_audio_file, FFT_HOP, AUDIO_N_SAMPLES, ANNOTATIONS_FPS, AUDIO_SAMPLE_RATE
from transformers import AutoFeatureExtractor
import numpy as np

class FeatureExtractorTorch:
    def __init__(self, sample_rate=16000, frame_rate=250):
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.spectrogram_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.basic_pitch_model = Model(ICASSP_2022_MODEL_PATH)
        return
    def compute_loudness_torch(self, audio,
                   n_fft=512,
                   range_db=80,
                   ref_db=0.0,
                   padding='constant'):
            """Perceptual loudness (weighted power) in dB using PyTorch.
            Adopted from DDSP: https://github.com/magenta/ddsp/blob/main/ddsp/spectral_ops.py
            Args:
            audio: Audio input that can be converted to a PyTorch tensor. Shape [batch_size, audio_length] or [audio_length,].
            sample_rate: Audio sample rate in Hz.
            frame_rate: Rate of loudness frames in Hz.
            n_fft: FFT window size.
            range_db: Sets the dynamic range of loudness in decibels.
            ref_db: Sets the reference maximum perceptual loudness.
            padding: padding mode for torch.nn.functional.pad.: 'constant', 'reflect', 'replicate', 'circular'. Default is 'constant'.

            Returns:
            Loudness in decibels. Shape [batch_size, n_frames] or [n_frames,].
            """
            # Convert audio to torch tensor if it isn't already
            if not isinstance(audio, torch.Tensor):
                audio = torch.tensor(audio, dtype=torch.float32)
            
            # Ensure audio is a 2D tensor
            if len(audio.shape) == 1:
              audio = audio.unsqueeze(0)

            # Pad audio
            hop_size = self.sample_rate // self.frame_rate
            pad_amount = (n_fft - hop_size) // 2
            audio = torch.nn.functional.pad(audio, (pad_amount, pad_amount), mode=padding)

            # Compute STFT
            stft = torch.stft(audio, n_fft=n_fft, hop_length=hop_size, win_length=n_fft, return_complex=True)
            power = stft.abs() ** 2

            # Perceptual weighting
            frequencies = torch.linspace(0, self.sample_rate // 2, n_fft // 2 + 1)
            a_weighting = torch.tensor(librosa.A_weighting(frequencies.numpy()), device=audio.device)
            a_weighting = 10 ** (a_weighting / 10)
            power = power
            a_weighting = a_weighting.unsqueeze(-1)
            power = a_weighting * power
            # Average over frequencies (weighted power per bin)
            avg_power = power.mean(dim=-1)
            loudness = 10 * torch.log10(avg_power + 1e-10)  # Convert to dB

            # Normalize loudness
            loudness = loudness - ref_db
            loudness = torch.clamp(loudness, min=-range_db)

            return loudness
    def preprocess_f0(self, audio,overlap_len, hop_size, original_length):
        audio_original = np.concatenate([np.zeros((int(overlap_len / 2),), dtype=np.float32), audio])
        for window, window_time in window_audio_file(audio_original, hop_size):
            yield np.expand_dims(window, axis=0), window_time, original_length
    
    def unwrap_output(self, output, audio_original_length: int, n_overlapping_frames: int,
    ) -> np.array:
        """Unwrap batched model predictions to a single matrix.

        Args:
            output: array (n_batches, n_times_short, n_freqs)
            audio_original_length: length of original audio signal (in samples)
            n_overlapping_frames: number of overlapping frames in the output

        Returns:
            array (n_times, n_freqs)
        """
        if len(output.shape) != 3:
            return None

        n_olap = int(0.5 * n_overlapping_frames)
        if n_olap > 0:
            # remove half of the overlapping frames from beginning and end
            output = output[:, n_olap:-n_olap, :]

        output_shape = output.shape
        n_output_frames_original = int(np.floor(audio_original_length * (ANNOTATIONS_FPS / AUDIO_SAMPLE_RATE)))
        unwrapped_output = output.reshape(output_shape[0] * output_shape[1], output_shape[2])
        return unwrapped_output[:n_output_frames_original, :]

    def get_f0(self, audio):
        # Preprocess audio
        n_overlapping_frames = 30
        overlap_len = n_overlapping_frames * FFT_HOP
        hop_size = AUDIO_N_SAMPLES - overlap_len
        original_length = audio.shape[0]
        output = {"note": [], "onset": [], "contour": []}
        for audio_windowed, _, audio_original_length in self.preprocess_f0(audio, overlap_len, hop_size, original_length):
            for k, v in self.basic_pitch_model.predict(audio_windowed).items():
                output[k].append(v)
        unwrapped_output = {
            k: self.unwrap_output(np.concatenate(output[k]), audio_original_length, n_overlapping_frames) for k in output
        }
        return torch.tensor(unwrapped_output['note'])
    
    def get_spectrogram(self, audio):
        spec = self.spectrogram_extractor(audio,self.sample_rate,return_tensors='pt')
        return spec['input_values']

    def get_features(self, audio):
        spec = self.get_spectrogram(audio)
        loudness = self.compute_loudness_torch(audio)
        f0 = self.get_f0(audio)
        return {"spectrogram":spec,"loudness":loudness,"f0":f0}