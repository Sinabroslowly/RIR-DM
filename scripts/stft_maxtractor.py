import numpy
import torch
import librosa
import torch.nn.functional as F

class STFT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._eps = 1e-8

    def transform(self, audio):
        #m = numpy.abs(librosa.stft(audio/numpy.abs(audio).max(), n_fft= 1024, hop_length=256))[:-1,:]
        m = numpy.abs(librosa.stft(audio, n_fft= 1024, hop_length=256))[:-1,:]
        #print(f"shape of spec: {m.shape}")
        m = numpy.log10(m + self._eps)
        m = (((m - m.min())/(m.max() - m.min()) * 2) - 1)
        #return (torch.FloatTensor if torch.cuda.is_available() else torch.Tensor)(m * 0.8).unsqueeze(0)
        return (torch.FloatTensor if torch.cuda.is_available() else torch.Tensor)(m).unsqueeze(0)
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

    def inverse(self, spec, maxtractor):
        #print(f"Shape of spec: {spec.shape}")
        s = spec.cpu().detach().numpy()
        #s_ori = numpy.exp((((s + 1) * 0.5) * 19.5) - 17.5) - self._eps # Empirical (average) min and max over test set
        spec_224 = F.interpolate(spec, size=(224,224), mode='bilinear', align_corners=False)
        s_factor = maxtractor.forward(spec_224) # Denormalize the generated spectrogram.
        #print(f"Shape of s_factor: {s_factor.shape}")
        s_max = s_factor[:, 0].view(-1, 1, 1, 1)  # Shape [batch, 1, 1, 1]
        s_min = s_factor[:, 1].view(-1, 1, 1, 1)  # Shape [batch, 1, 1, 1]

        # Expand s_max and s_min to match the shape of spec
        s_max = s_max.expand(-1, 1, 512, 512)  # Shape [batch, 1, 512, 512]
        s_min = s_min.expand(-1, 1, 512, 512)  # Shape [batch, 1, 512, 512]
        
        spec_denorm = numpy.power(10, (s + 1) * (s_max - s_min) / 2 + s_min) - self._eps
        # print(f"Max value of spec: {torch.max(spec)}")
        # print("Max value: ", s_max)
        # print(f"Max value of spec_ori: {numpy.max(s_ori)}")
        # print(f"Max value of spec_denorm: {numpy.max(spec_denorm)}")

        rp = numpy.random.uniform(-numpy.pi, numpy.pi, s.shape)
        f = spec_denorm * (numpy.cos(rp) + (1.j * numpy.sin(rp)))
        print(f"Shape of f: {f.shape}")
        #y = librosa.istft(f) # Reconstruct audio
        # return y
        return librosa.istft(f)
    '''

    def inverse(self, spec, maxtractor):
        s = spec  # Keep on GPU
        #print(f"Shape of s: {s.shape}")
        spec_224 = F.interpolate(spec, size=(224,224), mode='bilinear', align_corners=False)
        
        s_factor = maxtractor.forward(spec_224)  # Assume this is a PyTorch tensor on the GPU
        s_max = s_factor[:, 0].view(-1, 1, 1, 1)  # Shape [batch, 1, 1, 1]
        s_min = s_factor[:, 1].view(-1, 1, 1, 1)  # Shape [batch, 1, 1, 1]

        # Expand s_max and s_min to match the shape of spec
        s_max = s_max.expand(-1, 1, 512, 512)  # Shape [batch, 1, 512, 512]
        s_min = s_min.expand(-1, 1, 512, 512)  # Shape [batch, 1, 512, 512]
        #print(f"Shape of s_max: {s_max.shape} and Shape of s_min: {s_min.shape}")
        
        # Denormalize the spectrogram using PyTorch operations
        spec_denorm = torch.pow(10, (s + 1) * (s_max - s_min) / 2 + s_min) - self._eps

        # Generate random phase on GPU
        rp = torch.rand_like(spec) * (2 * torch.pi) - torch.pi
        f = spec_denorm * (torch.cos(rp) + (1.j * torch.sin(rp)))
        f = numpy.complex64(f.detach().cpu())

        # Perform ISTFT with torch.istft (if using PyTorch for ISTFT)
        #return torch.istft(f, n_fft=1024, hop_length=256)  # Adjust n_fft and hop_length as needed
        return librosa.istft(f)
