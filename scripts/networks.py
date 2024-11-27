import os
import numpy
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

class Maxtractor(nn.Module):
    def __init__(self, device="cuda", train_enc=True, dropout_prob = 0.3):
        super(Maxtractor, self).__init__()
        # Define separate ResNet for RGB and depth
        #self.rgb_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        #self.depth_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model = Classifier_Max(device=device, train_enc=train_enc)

        self.cnn = nn.Sequential(
            # The input is [batch_size, 2048, 1]
            nn.Conv1d(in_channels=1000, out_channels=256, kernel_size=3, padding=1),  # Output shape: [batch_size, 1024, 1]
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1),   # Output shape: [batch_size, 512, 1]
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1),    # Output shape: [batch_size, 365, 1]
            nn.AdaptiveAvgPool1d(1),  # This ensures the output is [batch_size, 365]
            nn.Flatten()  # Flatten the output to [batch_size, 365]
        )


        # Common final fully connected layer
        self.fc = nn.Linear(128, 2)  # Concatenating the RGB and depth features
        self.dropout = nn.Dropout(p=dropout_prob)

        self.model.to(device)

        if train_enc:
            self.model.train()
        else:
            self.model.eval()

    def forward(self, spec_input):
        features = self.model(spec_input)
        features = self.cnn(features.unsqueeze(2))
        features = self.dropout(features)

        x = self.fc(features)
        #x = torch.sigmoid(x) * 200  # Scale to range [0, 200]
        #x = torch.nn.functional.relu(x)  # Scale to range [0, 200]
        return x


class Classifier_Max(nn.Module):
    """
    Load encoder from pre-trained ResNet50 (Places365 CNN) model.
    """
    def __init__(self, device="cuda", train_enc=True):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1), # [batch, 3, 512, 512]
            nn.ReLU(),
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # [batch, 64, 256, 256]
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=2, padding=1),  # [batch, 3, 128, 128]
            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)  # [batch, 3, 224, 224]
        )
        self.model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        #self.model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        # self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        self.model.fc = nn.Identity()

        if train_enc:
            self.model.train()
        else:
            self.model.eval()

    def forward(self, x):
        #print(f"Shape of x after channel_conv: {x.shape}")
        x = self.downsample(x)
        x = self.model.forward(x)
        #print(f"Shape of x after model.forward: {x.shape}")
        return x
    
class FeatureMapGenerator(nn.Module):
    def __init__(self, image_size=512):
        super(FeatureMapGenerator, self).__init__()
        self.fc = nn.Linear(image_size, image_size*image_size)
        self.attn_layer = nn.MultiheadAttention(embed_dim=image_size, num_heads=8, batch_first=True)

    def forward(self, image_embedding, text_embedding):
        # Query, Key, Value
        Q, K, V = image_embedding.unsqueeze(0), text_embedding.unsqueeze(0), text_embedding.unsqueeze(0)
        x, _ = self.attn_layer(Q, K, V)
        x = x.squeeze(0)
        x = self.fc(x)
        x = x.view(-1, 512, 512)
        return x.unsqueeze(1)
    
############ Define ModulationLayer that is inspired from StyleGAN ############
class ModulationLayer(nn.Module):
    def __init__(self, embedding_dim, noise_channels):
        super(ModulationLayer, self).__init__()
        # Map the global embedding to noise channels for spatial scale and bias
        self.fc_scale = nn.Linear(embedding_dim, noise_channels)  # For scaling factor
        self.fc_bias = nn.Linear(embedding_dim, noise_channels)   # For bias
        self.expand = nn.Conv2d(noise_channels, noise_channels, kernel_size=1)  # Spatial expansion

    def forward(self, shared_embedding, noise):
        # Compute global scale and bias
        scale = self.fc_scale(shared_embedding).unsqueeze(-1).unsqueeze(-1)  # Shape: [batch, noise_channels, 1, 1]
        bias = self.fc_bias(shared_embedding).unsqueeze(-1).unsqueeze(-1)    # Shape: [batch, noise_channels, 1, 1]
        
        # Expand scale and bias to match spatial dimensions
        scale = self.expand(scale)  # Shape: [batch, noise_channels, 512, 512]
        bias = self.expand(bias)    # Shape: [batch, noise_channels, 512, 512]
        
        # Apply modulation
        return scale * noise + bias

