from pynvml import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from diffusers import DDIMScheduler, Transformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput

class FeatureMapGenerator(nn.Module):
    def __init__(self):
        super(FeatureMapGenerator, self).__init__()  # Initialize parent class

    def forward(self, text_embedding, image_embedding):
        # Compute outer product to generate shared embedding
        shared_embedding = torch.einsum('bi,bj->bij', text_embedding, image_embedding)  # [batch, 512, 512]
        shared_embedding = F.normalize(shared_embedding)  # Add channel dimension [batch, 512, 512]

        return shared_embedding.unsqueeze(1)

class LDT(nn.Module):
    def __init__(
        self,
        sample_size=64, # latent size of VQ-VAE representation (Five downsampling: 512-256-128-64-32)
        in_channels=8, # Dimension of VQ-VAE latent representation.
        #out_channels=8, # Dimension of VQ-VAE latent representation. Specify only if the output channel vaires from input channel.
        cross_attention_dim=128, # Added Reduced from 512 to 128 due to spatial compression.
        num_layers=2, # Default 18
        dropout=0.2,
        attention_head_dim=64, # Default 64
        num_attention_heads = 16, # Default 18
        activation_fn = "geglu",
        attention_bias = False,
        norm_num_groups = 4,
        num_train_timesteps=1000,
    ):
        super().__init__()

        self.transformer = Transformer2DModel(
            sample_size=sample_size,
            in_channels=in_channels,
            num_layers=num_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            cross_attention_dim=cross_attention_dim,
            dropout=dropout,
            activation_fn=activation_fn,
            attention_bias=attention_bias,
            norm_num_groups=norm_num_groups,
        )
        self.num_blocks = num_layers
        self.inner_dim = num_attention_heads * attention_head_dim

        self.spatial_compressor = nn.Sequential(
          nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False),
          nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, stride=2, padding=1, bias=False),
        )

        #self.cross_modal_embedding = nn.Linear(cross_attention_dim * cross_attention_dim, cross_attention_dim)
        self.cross_modal_proj = nn.Sequential(
            nn.Linear(1, cross_attention_dim),
            nn.LayerNorm(cross_attention_dim),
        )


        self.scheduler = DDIMScheduler(beta_start = 0.0001, 
                                                        beta_end=0.02,
                                                        beta_schedule='linear',
                                                        clip_sample_range=0.8,
                                                        prediction_type='epsilon',
                                                        num_train_timesteps=num_train_timesteps) # Noise scheduler

    def forward(self, latent_input, cross_modal_embedding=None, timestep=None, unconditioned=None, training=True, return_dict=True):
      """
      Forward function with random null embedding replacement for unconstrained generation.

      Args:
          latent_input (torch.FloatTensor): Input latent representations.
          cross_modal_embedding (torch.FloatTensor, optional): Cross-modal embeddings (e.g., captions or image features).
          timestep (torch.LongTensor, optional): Timestep information for denoising.
          p_uncon (float): Probability of replacing embeddings with random null embeddings.
          training (bool): Whether the model is in training mode.

      Returns:
          torch.FloatTensor: Denoised output.
      """
      pooled_projection = None

      if cross_modal_embedding is not None:
          # Spatial compression
          embedding = self.spatial_compressor(cross_modal_embedding)
          embedding = nn.LayerNorm(embedding.shape[1:]).to(embedding.device)(embedding)
          #print(f"Shape of cross_modal_embedding after compression: {embedding.shape}")  # [4, 1, 128, 128]

          # Flatten for projection
          embedding = embedding.view(embedding.shape[0], -1, 1)
          #print(f"Shape of cross_modal_embedding after flattening: {embedding.shape}")  # [4, 16384, 1]

          # Random null embedding replacement (only during training)
          if unconditioned is not None and training:
              random_null_embedding = torch.randn_like(embedding)
              random_null_embedding = F.normalize(random_null_embedding, dim=-1)
              # mask = torch.rand(embedding.size(0), device=embedding.device) < p_uncon
              # mask = mask.view(-1, 1, 1)  # Broadcast mask
              # embedding = torch.where(mask, random_null_embedding, embedding)
              embedding = torch.where(
                unconditioned.view(-1, 1, 1),
                random_null_embedding,
                embedding
              )

          # Cross-modal projection
          embedding = self.cross_modal_proj(embedding)
          #print(f"Shape of cross_modal_embedding after projection: {embedding.shape}")  # [4, 16384, 128]

      # Pass through the transformer
      denoised_output = self.transformer(
          hidden_states=latent_input,
          encoder_hidden_states=embedding if cross_modal_embedding is not None else None,
          timestep=timestep,
          cross_attention_kwargs=None,
          return_dict=return_dict,
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
    noise_channels = 8
    image_size = 64
    embedding_size = 512
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Initialize the model
    model = LDT().to(device)
    featuremap_generator = FeatureMapGenerator().to(device)

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

    print("Denoised output shape:", denoised_output.shape) # [batch, noise_channels, 64, 64]


if __name__ == "__main__":
    main()