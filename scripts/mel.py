import numpy
import torch
import librosa


class LogMel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._eps = 1e-8

    def transform(self, audio):
        #print(f"Shape of audio: {audio.shape}")
        #m = librosa.feature.melspectrogram(audio/numpy.abs(audio).max())
        m = librosa.feature.melspectrogram(y = audio, sr=24000, n_mels = 100, n_fft=1024, hop_length=256, win_length=1024)

        # Zero-pad the spectrogram to match 520 length temporal resolution
        if m.shape[1] < 520:
            m = numpy.pad(m, ((0, 0), (0, 520 - m.shape[1])), mode='constant')
        else:
            m = m[1:520]

        m = numpy.log(m + self._eps)
        #print(f"Shape of the mel-spectrogram: {m.shape}")
        return torch.Tensor(((m - m.mean()) / m.std()) * 0.8).unsqueeze(0)

    def inverse(self, spec):
        s = spec.cpu().detach().numpy()
        s = numpy.exp((s * 5) - 15.96) - self._eps # Empirical mean and standard deviation over test set
        y = librosa.feature.inverse.mel_to_audio(s) # Reconstruct audio
        return y/numpy.abs(y).max()
