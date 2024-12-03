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

class LDT(nn.Module):
    def __init__(
        self,
        sample_size=32, # latent size of VQ-VAE representation (Five downsampling: 512-256-128-64-32)
        in_channels=16, # Dimension of VQ-VAE latent representation.
        out_channels=16, # Dimension of VQ-VAE latent representation.
        cross_attention_dim=512, # Added
        num_layers=6, # Default 18
        attention_head_dim=64, # Default 64
        num_attention_heads = 8, # Default 18
        joint_attention_dim=512, # Default 4096
        patch_size=2, # Default 2
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
        )
        self.num_blocks = num_layers
        self.inner_dim = num_attention_heads * attention_head_dim

        self.cross_modal_embedding = nn.Linear(cross_attention_dim, joint_attention_dim)

        self.controlnet_proj = nn.Linear(cross_attention_dim, self.num_blocks * self.inner_dim)

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
          print(f"Shape of cross_modal_embedding after projection: {cross_modal_embedding.shape}")

          control_states = self.controlnet_proj(cross_modal_embedding)
          print(f"Shape of control_states after projection: {control_states.shape}")
          control_states = control_states.view(latent_input.shape[0],self.num_blocks, self.inner_dim, latent_input.shape[2], latent_input.shape[3])

          block_controlnet_hidden_states = torch.split(control_states, 1, dim=1)
        else:
          block_controlnet_hidden_states = None

        denoised_output = self.transformer(
            hidden_states=latent_input,
            encoder_hidden_states=cross_modal_embedding,
            timestep=timestep,
            block_controlnet_hidden_states=block_controlnet_hidden_states,
            joint_attention_kwargs = None,
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
    print(f"Shape of the latent_input_test: {latent_input_test.shape}")

    # Condition tensor: [batch, conditional_channels, 512, 512] (feature map from shared embedding)
    image_embedding = torch.randn(batch_size, embedding_size).to(device)
    text_embedding = torch.randn(batch_size, embedding_size).to(device)

    cross_modal_embedding = featuremap_generator(text_embedding, image_embedding)
    print(f"Shape of the cross_modal_embedding: {cross_modal_embedding.shape}")

    # Timestep (current diffusion timestep)
    timestep = torch.randint(0, model.scheduler.config.num_train_timesteps, (batch_size,), device=latent_input_test.device)

    # Forward pass
    denoised_output = model(latent_input_test, cross_modal_embedding, timestep)

    print_gpu_utilization()

    print("Denoised output shape:", denoised_output.shape) # [batch, noise_channels, 512, 512]


if __name__ == "__main__":
    main()
