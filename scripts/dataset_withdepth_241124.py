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
    ".png", ".PNG", ".ppm", ".PPM", ".bmp", ".BMP", ".tiff", ".wav", ".WAV", ".aif", ".aiff", ".AIF", ".AIFF", ".pt", ".json"
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


class Image2ReverbDataset(Dataset):
    def __init__(self, dataroot, device, model= None, transform= None, phase="train", spec="stft"):
        self.device = device
        self.phase = phase
        self.root = dataroot
        self.stft = LogMel() if spec == "mel" else STFT()
        self.depth_model = model
        self.transform = transform

        ### input A (images)
        dir_A = "_A"
        self.dir_A = os.path.join(self.root, phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        ### input B (audio)
        dir_B = "_B"
        self.dir_B = os.path.join(self.root, phase + dir_B)  
        self.B_paths = sorted(make_dataset(self.dir_B))

        ### input_C (depth)
        if phase != "test":
            dir_E = "_E"
            self.dir_C = os.path.join(self.root, phase + dir_E)
            self.C_paths = sorted(make_dataset(self.dir_E))
      
    def __getitem__(self, index):
        if index > len(self):
            return None
        ### input A (images)
        A_path = self.A_paths[index]              
        A = Image.open(A_path)
        t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        A_tensor = t(A.convert("RGB"))
        width, height = A.size
        min_dim = min(width, height)
        A_tensor = transforms.functional.center_crop(A_tensor, min_dim)
        A_tensor = transforms.functional.resize(A_tensor, 224)
        #print(f"Shape of the transformed t: {A_tensor.shape}")

        ### input B (audio)
        B_path = self.B_paths[index]
        B, _ = soundfile.read(B_path)
        B_spec = self.stft.transform(B)

        ### input C (depth pt)
        if self.phase != "test":
            C_path = self.C_paths[index]
            #print(f"C_path: {C_path}")
            C = torch.load(C_path)
            if C.dim() == 2:
                C = C.unsqueeze(0)
            #print(f"Shape of C: {C.shape}")
            _, width, height = C.shape
            C_tensor = transforms.functional.center_crop(C, min_dim)
            #C_tensor = transforms.functional.resize(C_tensor, 224)
            C_tensor = F.interpolate(C_tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
            #print(f"Shape of C after crop: {C_tensor.shape}")

            return B_spec, A_tensor, C_tensor, (B_path, A_path, C_path)

            #return B_spec, A_tensor, (B_path, A_path) # original where we don't use explicitly generated depth image during the train phase.
        else:
            if self.depth_model != None:
                A_trans = cv2.cvtColor(np.array(A), cv2.COLOR_RGB2BGR)
                with torch.no_grad():
                    A_trans = self.transform(A_trans).to(self.device)
                    C = self.depth_model(A_trans)
                    C = torch.nn.functional.interpolate(
                        C.unsqueeze(1),
                        size=A_trans.shape[-2:],
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze()

                if C.dim() == 2:
                    C = C.unsqueeze(0)
                _, width, height = C.shape
                min_dim = min(width, height)
                C_tensor = transforms.functional.center_crop(C, min_dim)
                #C_tensor = transforms.functional.resize(C_tensor, 224)
                C_tensor = F.interpolate(C_tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)

                return B_spec, A_tensor, C_tensor, (B_path, A_path)
                


        

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return "Image2Reverb"
