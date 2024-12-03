from pynvml import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import EDMDPMSolverMultistepScheduler, SD3Transformer2DModel
from diffusers.configuration_utils import ConfigMixIn, register_to_config
from diffusers.modeling_utils import ModelMixIn

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
        num_train_timesteps=1000,
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

        self.scheduler = EDMDPMSolverMultistepScheduler(sigma_min=0.002, 
                                                        sigma_max=80.0, 
                                                        sigma_data=0.5,
                                                        sigma_schedule='karras',
                                                        solver_order=2,
                                                        prediction_type='epsilon',
                                                        num_train_timesteps=num_train_timesteps) # Noise scheduler

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
    batch_size = 4
    noise_channels = 16
    image_size = 32
    embedding_size = 512
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Initialize the model
    model = LDT().to(device)
    featuremap_generator = FeatureMapGenerator()

    print(f"The model has {sum(p.numel() for p in model.parameters())} parameters")

    # Noise tensor (noisy spectrogram)
    latent_input_test = torch.randn(batch_size, noise_channels, image_size, image_size).to(device)

    # Condition tensor: [batch, conditional_channels, 512, 512] (feature map from shared embedding)
    image_embedding = torch.randn(batch_size, embedding_size).to(device)
    text_embedding = torch.randn(batch_size, embedding_size).to(device)

    cross_modal_embedding = featuremap_generator(text_embedding, image_embedding)

    # Timestep (current diffusion timestep)
    timestep = torch.randint(0, model.scheduler.config.num_train_timesteps, (batch_size,), device=latent_input_test.device)

    # Forward pass
    denoised_output = model(latent_input_test, cross_modal_embedding, timestep)

    print_gpu_utilization()

    print("Denoised output shape:", denoised_output.shape) # [batch, noise_channels, 512, 512]


if __name__ == "__main__":
    main()
