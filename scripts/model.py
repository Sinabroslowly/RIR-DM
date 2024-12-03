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

        return shared_embedding.unsqueeze(1)

class LDT(nn.Module):
    def __init__(
        self,
        sample_size=32, # latent size of VQ-VAE representation (Five downsampling: 512-256-128-64-32)
        in_channels=16, # Dimension of VQ-VAE latent representation.
        out_channels=16, # Dimension of VQ-VAE latent representation.
        cross_attention_dim=128, # Added Reduced from 512 to 128 due to spatial compression.
        num_layers=12, # Default 18
        attention_head_dim=64, # Default 64
        num_attention_heads = 8, # Default 18
        sequence_dim=1024, # Added
        joint_attention_dim=1152, # Default 4096
        pooled_projection_dim=2048, # Default 2048
        caption_projection_dim=512,
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
            pooled_projection_dim=pooled_projection_dim,
            caption_projection_dim=caption_projection_dim,
        )
        self.num_blocks = num_layers
        self.inner_dim = num_attention_heads * attention_head_dim
        self.sequence_dim = sequence_dim,
        self.joint_attention_dim = joint_attention_dim

        self.spatial_compressor = nn.Sequential(
          nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False),
          nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, stride=2, padding=1, bias=False),
        )

        self.cross_modal_embedding = nn.Linear(cross_attention_dim * cross_attention_dim, sequence_dim)
        self.cross_modal_proj = nn.Sequential(
            nn.Linear(1, joint_attention_dim),
            nn.LayerNorm(joint_attention_dim),
        )
        self.pooled_proj = nn.Sequential(
            nn.Linear(joint_attention_dim, pooled_projection_dim),
            nn.LayerNorm(pooled_projection_dim),
        )

        self.scheduler = EDMDPMSolverMultistepScheduler(sigma_min=0.002, 
                                                        sigma_max=80.0, 
                                                        sigma_data=0.5,
                                                        sigma_schedule='karras',
                                                        solver_order=2,
                                                        prediction_type='epsilon',
                                                        num_train_timesteps=num_train_timesteps) # Noise scheduler

    def forward(self, latent_input, cross_modal_embedding=None, timestep=None, p_uncon=0.2, training=True):
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
        embedding = nn.LayerNorm(embedding.size()[1:])(embedding)
        print(f"Shape of cross_modal_embedding after compression: {embedding.shape}")  # [4, 1, 128, 128]

        # Flatten for projection
        embedding = embedding.view(embedding.shape[0], -1, 1)
        print(f"Shape of cross_modal_embedding after flattening: {embedding.shape}")  # [4, 16384, 1]

        # Random null embedding replacement (only during training)
        if training and p_uncon > 0:
            random_null_embedding = torch.randn_like(embedding)
            random_null_embedding = F.normalize(random_null_embedding, dim=-1)
            mask = torch.rand(embedding.size(0), device=embedding.device) < p_uncon
            mask = mask.view(-1, 1, 1)  # Broadcast mask
            embedding = torch.where(mask, random_null_embedding, embedding)

        # Cross-modal projection
        embedding = self.cross_modal_proj(embedding)
        print(f"Shape of cross_modal_embedding after projection: {embedding.shape}")  # [4, 16384, 4096]

        # Pooled projection
        pooled_projection = self.pooled_proj(embedding.mean(dim=1))
        print(f"Shape of pooled_projection: {pooled_projection.shape}")  # [4, 2048]

    # Pass through the transformer
    denoised_output = self.transformer(
        hidden_states=latent_input,
        encoder_hidden_states=embedding if cross_modal_embedding is not None else None,
        timestep=timestep,
        pooled_projections=pooled_projection,
        joint_attention_kwargs=None,
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
