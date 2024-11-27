import numpy
import torch
import librosa


class STFT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._eps = 1e-8

    def transform(self, audio):
        m = numpy.abs(librosa.stft(audio/numpy.abs(audio).max(), n_fft= 1024, hop_length=256))[:-1,:]
        #print(f"shape of spec: {m.shape}")
        m = numpy.log(m + self._eps)
        m = (((m - m.min())/(m.max() - m.min()) * 2) - 1)
        return (torch.FloatTensor if torch.cuda.is_available() else torch.Tensor)(m * 0.8).unsqueeze(0)
    '''
    def transform(self, audio):
        m = numpy.abs(librosa.stft(audio, n_fft=1024, hop_length=256))[:-1,:]
        target_shape = (m.shape[0], m.shape[0])
        target_matrix = numpy.zeros(target_shape)
        if m.shape[1] > m.shape[0]:
            target_matrix = m[:,:m.shape[0]] # Crop out the lengthy part of the spectrogram to fit 512 x 512 shape.
        else:
            target_matrix[:, :m.shape[1]] = m
        #print(f"Shape of target_matrix: {target_matrix.shape}")
        #target_matrix[:, :m.shape[1]] = m
        m = numpy.log(target_matrix + self._eps)
        m = (((m - m.min())/(m.max() - m.min()) * 2) - 1)
        return (torch.FloatTensor if torch.cuda.is_available() else torch.Tensor)(m * 0.8).unsqueeze(0)
    '''
    def inverse(self, spec):
        s = spec.cpu().detach().numpy()
        s = numpy.exp((((s + 1) * 0.5) * 19.5) - 17.5 - self._eps) # Empirical (average) min and max over test set
        rp = numpy.random.uniform(-numpy.pi, numpy.pi, s.shape)
        f = s * (numpy.cos(rp) + (1.j * numpy.sin(rp)))
        y = librosa.istft(f) # Reconstruct audio
        return y/numpy.abs(y).max()
