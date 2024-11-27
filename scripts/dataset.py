import os
import soundfile
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torch.utils.data import Dataset
from PIL import Image
from .stft import STFT
from .mel import LogMel


F_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG",
    ".png", ".PNG", ".ppm", ".PPM", ".bmp", ".BMP", ".tiff", ".wav", ".WAV", ".aif", ".aiff", ".AIF", ".AIFF", ".pt"
]


def is_image_audio_file(filename):
    return any(filename.endswith(extension) for extension in F_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), "%s is not a valid directory." % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_audio_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


class RIRDDMDataset(Dataset):
    def __init__(self, dataroot, device, model= None, transform= None, phase="train", spec="stft"):
        self.device = device
        self.phase = phase
        self.root = dataroot
        self.stft = LogMel() if spec == "mel" else STFT()
        self.depth_model = model
        self.transform = transform

        ### input B (audio)
        dir_B = "_B"
        self.dir_B = os.path.join(self.root, phase + dir_B)  
        self.B_paths = sorted(make_dataset(self.dir_B))

        ### input E (CLIP embedding)
        dir_E = "_E"
        self.dir_E = os.path.join(self.root, phase + dir_E)
        self.E_paths = sorted(make_dataset(self.dir_E))
      
    def __getitem__(self, index):
        if index > len(self):
            return None
        ### input B (audio)
        B_path = self.B_paths[index]
        B, _ = soundfile.read(B_path)
        B_spec = self.stft.transform(B)

        ### input E (clip embedding, [2, 512])
        E_path = self.E_paths[index]
        #print(f"C_path: {C_path}")
        E_embedding = torch.load(E_path, weights_only=True)
        #print(f"Index: {index}, spectrogram: {B_spec.shape}, embeddings: {E_embedding.shape}")

        return B_spec.detach(), E_embedding[0].detach(), E_embedding[1].detach(), (B_path, E_path) 

    def __len__(self):
        return len(self.B_paths)

    def name(self):
        return "RIRLDM"
