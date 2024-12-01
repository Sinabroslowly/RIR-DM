import os
import math
import numpy
import torch
import torch.fft
import torch.nn.functional as F
import soundfile as sf
from PIL import Image


def hilbert(x): #hilbert transform
    N = x.shape[1]
    Xf = torch.fft.fft(x, n=None, dim=-1)
    h = torch.zeros(N)
    if N % 2 == 0:
        h[0] = h[N//2] = 1
        h[1:N//2] = 2
    else:
        h[0] = 1
        h[1:(N + 1)//2] = 2
    x = torch.fft.ifft(Xf * h)
    return x


def spectral_centroid(x): #calculate the spectral centroid "brightness" of an audio input
    Xf = torch.abs(torch.fft.fft(x,n=None,dim=-1)) #take fft and abs of x
    norm_Xf = Xf / sum(sum(Xf))  # like probability mass function
    norm_freqs = torch.linspace(0, 1, Xf.shape[1])
    spectral_centroid = sum(sum(norm_freqs * norm_Xf))
    return spectral_centroid


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=numpy.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (numpy.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = numpy.transpose(image_numpy, (1, 2, 0)) * 255.0      
    image_numpy = numpy.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:        
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)

# Converts a one-hot tensor into a colorful label map
def tensor2label(label_tensor, n_label, imtype=numpy.uint8):
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()    
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = numpy.transpose(label_tensor.numpy(), (1, 2, 0))
    return label_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

################### This part is the custom part that is different from the original Image2Reverb code. #########################

def calculate_mask(a, b, comparison_operator):
    return comparison_operator(a, b) #operator.gt, operator.lt, operator.ge, operator.le

def calculate_active_region(spectrogram, comparison_operator, threshold=-60, lin_flag = False):
    # Convert dBFS to linear scale (assuming the spectrogram is in dBFS)
    threshold_linear = 10**(threshold / 20)
    if lin_flag:
        active_region = calculate_mask(spectrogram, threshold_linear, comparison_operator)
    else:
        active_region = calculate_mask(spectrogram, threshold, comparison_operator)

    return active_region

def calculate_iou(region1, region2):
    #intersection = numpy.sum(region1 & region2)
    intersection = torch.sum(torch.logical_and(region1, region2).float(), dim=[1, 2, 3])
    #union = numpy.sum(region1 | region2)
    union = torch.sum(torch.logical_or(region1, region2).float(), dim=[1, 2, 3])
    #return intersection / union if union != 0 else 0

    iou_score = intersection / (union + 1e-8)

    return iou_score.mean()


def calculate_weighted_iou_loss(generated_image, target_image, mask_threshold=-60, comparison_operator=torch.ge, loss_name='l2', alpha=1.0, beta=1.0):
    # Calculate masks for the active regions
    target_mask = calculate_active_region(target_image, comparison_operator, mask_threshold)
    generated_mask = calculate_active_region(generated_image, comparison_operator, mask_threshold)

    # Calculate the Intersection over Union for the active regions
    iou_score = calculate_iou(target_mask, generated_mask)    

    '''
    # l2_loss = F.mse_loss((generated_image - target_image) ** 2)
    if loss_name == 'l2':
        loss = F.mse_loss(generated_image, target_image)
    elif loss_name == 'l1':
        loss = F.l1_loss(generated_image, target_image)
    else:
        raise ValueError(f"Invalid loss name '{loss_name}'. Choose 'l2' or 'l1'.")
    
    weighted_loss = alpha * loss * (1 + beta * (1 - iou_score))
    '''
    weighted_loss = 1 - iou_score

    return weighted_loss

###############################################################################
# Code from
# https://github.com/ycszen/pytorch-seg/blob/master/transform.py
# Modified so it complies with the Citscape label map colors
###############################################################################
def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def labelcolormap(N):
    if N == 35: # cityscape
        cmap = numpy.array([(  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (111, 74,  0), ( 81,  0, 81),
                     (128, 64,128), (244, 35,232), (250,170,160), (230,150,140), ( 70, 70, 70), (102,102,156), (190,153,153),
                     (180,165,180), (150,100,100), (150,120, 90), (153,153,153), (153,153,153), (250,170, 30), (220,220,  0),
                     (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
                     (  0, 60,100), (  0,  0, 90), (  0,  0,110), (  0, 80,100), (  0,  0,230), (119, 11, 32), (  0,  0,142)], 
                     dtype=numpy.uint8)
    else:
        cmap = numpy.zeros((N, 3), dtype=numpy.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (numpy.uint8(str_id[-1]) << (7-j))
                g = g ^ (numpy.uint8(str_id[-2]) << (7-j))
                b = b ^ (numpy.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap

class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image
    

def compare_t60(a, b, sr=86): 
    # original sample rate / hop size (22050 / 256) = 86.1328125 
    # -> Changed to 48000 / 256 = 187.5 due to upsampled IR SR.
    try:
        a = a.detach().clone().abs()
        b = b.detach().clone().abs()
        a = (a - a.min())/(a.max() - a.min())
        b = (b - b.min())/(b.max() - b.min())
        t_a = estimate_t60(a, sr)
        t_b = estimate_t60(b, sr)
        return abs((t_b - t_a)/( t_a + 1e-8)) * 100
    except Exception as error:
        return 100


def estimate_t60(audio, sr):
    fs = float(sr)
    audio = audio.detach().clone()

    decay_db = 20

    # The power of the impulse response in dB
    power = audio ** 2
    energy = torch.flip(torch.cumsum(torch.flip(power, [0]), 0), [0])  # Integration according to Schroeder

    # remove the possibly all zero tail
    i_nz = torch.max(torch.where(energy > 0)[0])
    n = energy[:i_nz]
    db = 10 * torch.log10(n)
    db = db - db[0]

    # -5 dB headroom
    i_5db = torch.min(torch.where(-5 - db > 0)[0])
    e_5db = db[i_5db]
    t_5db = i_5db / fs

    # after decay
    i_decay = torch.min(torch.where(-5 - decay_db - db > 0)[0])
    t_decay = i_decay / fs

    # compute the decay time
    decay_time = t_decay - t_5db
    est_rt60 = (60 / decay_db) * decay_time

    return est_rt60

def get_octave_band_masks():
    # Frequency bin centers for a spectrogram of shape [512, 512] (n_fft=1024) with a sample rate of 86 Hz.
    #freqs = torch.linspace(0, sr // 2, n_fft // 2)  # Adjusted frequency resolution due to downsampled SR
    freqs = torch.linspace(0, 511, 512)
    
    # Central frequencies converted to frequency bin indices for the reduced set
    cfreqs = torch.tensor([6, 12, 23, 46, 93, 186])

    # Generate a list of masks for each octave band
    band_masks = []
    for fc in cfreqs:
        fl = torch.floor(fc / 2 ** (1 / 2))
        fh = torch.floor(fc * 2 ** (1 / 2))

        # Find the frequency bins that correspond to these cutoff frequencies
        band_mask = (freqs >= fl) & (freqs <= fh)
        band_masks.append(band_mask)

    return band_masks


def compare_t60_octave_bandwise(real_spec, fake_spec, sr=86):
    band_masks = get_octave_band_masks()

    #print(f"Shake of spec: {real_spec.shape}")

    band_t60_errors = []

    for band_mask in band_masks:
        # Apply the mask  to isolate the frequencies in the octave band
        real_band = real_spec[:, band_mask, :]
        fake_band = fake_spec[:, band_mask, :]

        # Sum the energies over the freq bins in each band (collapse to a single column vector)
        real_band_energy = torch.pow(10,real_band).sum(-2).squeeze()
        fake_band_energy = torch.pow(10,fake_band).sum(-2).squeeze()

        #print(f"Shape of real_band_energy: {real_band_energy.shape}")
        #print(f"Shape of fake_band_energy: {fake_band_energy.shape}")

        # Calculate T60 error for this octave band
        t60_err = compare_t60(real_band_energy +1e-8 , fake_band_energy + 1e-8, sr)
        band_t60_errors.append(t60_err)

    #band_t60_errors_cpu = band_t60_errors.cpu()

    # Calculate the T60 errors across all bands with a specific weighting.
    #avg_t60_err = weighted_t60_err(band_t60_errors)

    #print(f"T60 value error is: {band_t60_errors_cpu}")

    return band_t60_errors

def weighted_t60_err(band_t60_errors):
    # Later, define the function that will calculate the weighted error for each frequency bands.
    #t60_err = band_t60_errors.clone().detach().mean()
    t60_err = band_t60_errors.mean()
    #print(f"t60_err: {t60_err}")
    return t60_err

def main(audio_path, fake_audio_path):
    from stft import STFT
    # Load audio file using soundfile (sf)
    audio, sr = sf.read(audio_path)
    fake_audio, sr = sf.read(fake_audio_path)

    # Define the STFT parameters
    n_fft = 1024
    hop_size = 256
    window = torch.hann_window(n_fft)

    stft = STFT()

    # Calculate the spectrogram (real and imaginary components)
    real_spec = stft.transform(audio)
    fake_spec = stft.transform(fake_audio)

    # Calculate broadband T60 error
    sr_effective = round(sr / hop_size)  # e.g., 22050 / 256 = 86 Hz
    broadband_t60_err = compare_t60(
        torch.exp(real_spec).sum(-2).squeeze(),
        torch.exp(fake_spec).sum(-2).squeeze(),
        sr=sr_effective
    )
    print(f"Broadband T60 error: {broadband_t60_err}%")

    # Calculate octave-band T60 error
    octave_band_t60_errs = compare_t60_octave_bandwise(real_spec, fake_spec, sr=sr_effective)
    weighted_octave_t60_err = weighted_t60_err(octave_band_t60_errs)
    print(f"Octave-band weighted T60 error: {weighted_octave_t60_err}%")

## Test function
if __name__ == "__main__":
    # Path to the audio file (e.g., "audio_sample.wav")
    audio_path = "/home/airis_lab/MJ/Image2Reverb_scratch/test_IR.wav"
    fake_audio_path = "/home/airis_lab/MJ/Image2Reverb_scratch/test_IR_inferred.wav"
    main(audio_path, fake_audio_path)
