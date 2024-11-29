from pynvml import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DModel, UNet2DConditionModel
#from .networks import FeatureMapGenerator, ModulationLayer

class FeatureMapGenerator(nn.Module):
    def __init__(self):
        super(FeatureMapGenerator, self).__init__()  # Initialize parent class

    def forward(self, text_embedding, image_embedding):
        # Compute outer product to generate shared embedding
        shared_embedding = torch.einsum('bi,bj->bij', text_embedding, image_embedding)  # [batch, 512, 512]
        shared_embedding = F.normalize(shared_embedding)  # Add channel dimension [batch, 512, 512]

        return shared_embedding
    

class ConditionalDDPM(nn.Module):
    def __init__(self, noise_channels=1, condition_channels=1, embedding_dim=512, image_size=512, num_train_timesteps=1000):
        super().__init__()
        #self.feature_map_generator = FeatureMapGenerator(image_size=image_size)
        self.feature_map_generator = FeatureMapGenerator()

        self.unet = UNet2DModel(
            sample_size = image_size, # Image size
            in_channels = noise_channels + condition_channels, # Input channels = noise + conditional
            out_channels = noise_channels, # Output channels = denoised noise
            layers_per_block = 2, # Layers per UNet block
            #block_out_channels = (128, 128, 256, 256, 512, 512), # Output channels for each block
            block_out_channels = (128, 128, 256, 256, 512, 512),
            down_block_types = ("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D", "DownBlock2D"), # Down block types
            up_block_types = ("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"), # Up block types
            dropout = 0.2
        )

        self.scheduler = DDPMScheduler(num_train_timesteps) # Noise scheduler

    def forward(self, sample, timestep, text_embedding, image_embedding):
        # Generate condition with image_embedding and text_embedding
        encoder_hidden_states = self.feature_map_generator(text_embedding, image_embedding)
        #encoder_hidden_states = torch.cat([text_embedding.unsqueeze(-1), image_embedding.unsqueeze(-1)], dim=2)
        # Reshape (4, 512, 2) 

        # Concatenate noise and condition on the channel dimension
        return self.unet(torch.cat([sample, encoder_hidden_states.unsqueeze(1)], dim=1), timestep).sample # Output denoised noise

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

def main():
    # Example inputs
    batch_size = 1
    noise_channels = 1
    conditional_channels = 1
    image_size = 512
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Initialize the model
    model = ConditionalDDPM(noise_channels=1, conditional_channels=1, image_size=512).to(device)

    print(f"The model has {sum(p.numel() for p in model.parameters())} parameters")

    # Noise tensor (noisy spectrogram)
    noisy_spectrogram = torch.randn(batch_size, noise_channels, image_size, image_size).to(device)

    # Condition tensor: [batch, conditional_channels, 512, 512] (feature map from shared embedding)
    image_embedding = torch.randn(batch_size, image_size).to(device)
    text_embedding = torch.randn(batch_size, image_size).to(device)

    # Timestep (current diffusion timestep)
    timestep = torch.randint(0, model.scheduler.config.num_train_timesteps, (batch_size,), device=noisy_spectrogram.device)

    # Forward pass
    denoised_output = model(noisy_spectrogram, timestep, image_embedding, text_embedding)

    print_gpu_utilization()

    print("Denoised output shape:", denoised_output.shape) # [batch, noise_channels, 512, 512]


if __name__ == "__main__":
    main()
