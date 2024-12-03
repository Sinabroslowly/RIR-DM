from pynvml import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import EDMDPMSolverMultistepScheduler, SD3Transformer2DModel

class FeatureMapGenerator(nn.Module):
    def __init__(self):
        super(FeatureMapGenerator, self).__init__()  # Initialize parent class

    def forward(self, text_embedding, image_embedding):
        # Compute outer product to generate shared embedding
        shared_embedding = torch.einsum('bi,bj->bij', text_embedding, image_embedding)  # [batch, 512, 512]
        shared_embedding = F.normalize(shared_embedding)  # Add channel dimension [batch, 512, 512]

        return shared_embedding

class LDT(ModelMixIn, ConfigMixIn):
    @register_to_config
    def __init__(
        self,
        sample_size=32,
        in_channels=16,
        out_channels=16,
        cross_attention_dim=512,
        num_layers=18,
        attention_head_dim=64,
        num_attention_heads = 8,
        joint_attention_dim=4096,
        patch_size=2,
    ):
        super().__init__()

        self.transformer = SD3Transformer2DModel(
            sample_size=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            cross_attention_dim=cross_attention_dim,
        )

        self.cross_modal_embedding = nn.Linear(cross_attention_dim, joint_attention_dim)

    def forward(self, latent_input, cross_modal_embedding=None, timestep=None):
        if cross_modal_embedding is not None:
            cross_modal_embedding = self.cross_modal_embedding(cross_modal_embedding)

        denoised_output = self.transformer(
            hidden_states=latent_input,
            encoder_hidden_states=cross_modal_embedding,
            timestep=timestep,
        ).sample

        return denoised_output

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
    model = LDT().to(device)

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
