from diffusers import UNet2DModel
from .networks import Encoder, CrossAttention
import torch.nn as nn
import torch



class ConditionalUNet2DModel(UNet2DModel):
    def __init__(self, *args, condition_channels=365, num_heads=8, **kwargs):
        super().__init__(*args, **kwargs)
        #self.condition_conv = nn.Conv2d(condition_channels, 512, kernel_size=3, padding=1)
        # self.imagify = nn.Sequential(
        #     nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # (batch, 256, 2, 2)
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (batch, 128, 4, 4)
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # (batch, 64, 8, 8)
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # (batch, 32, 16, 16)
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),    # (batch, 16, 32, 32)
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),     # (batch, 8, 64, 64)
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1),      # (batch, 4, 128, 128)
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(4, 2, kernel_size=4, stride=2, padding=1),      # (batch, 2, 256, 256)
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(2, 1, kernel_size=4, stride=2, padding=1),      # (batch, 1, 512, 512)
        # )
        self.condition_proj = nn.Linear(condition_channels, 512)

        self.attn_down_blocks = nn.ModuleList([
            CrossAttention(embed_dim=512, num_heads=num_heads) for _ in range(2)
        ])
        self.attn_up_blocks = nn.ModuleList([
            CrossAttention(embed_dim=512, num_heads=num_heads) for _ in range(2)
        ])

    # def forward(self, noisy_images, timesteps, conditions, **kwargs):
    #     conditioned_input = self.condition_conv(conditions)
    #     noisy_images += conditioned_input
    #     return super().forward(noisy_images, timesteps, **kwargs)

    def forward(self, noisy_images, timesteps, conditions, **kwargs):
        # Project conditions to feature space
        B, C, H, W = noisy_images.shape
        cond = self.condition_proj(conditions.squeeze(-1).squeeze(-1)).view(B, -1, 1, 1)
        cond = cond.expand(-1, -1, H, W)

        # Pass through the U-Net Layers
        for i, layer in enumerate (self.encoder_blocks):
            noisy_images = layer(noisy_images) # Normal DownBlock2D or AttnDownBlock2D

            # Apply cross-attention for AttnDownBlock2D
            if isinstance(layer, CrossAttention) and i < len(self.attn_down_blocks):
                noisy_images = self.attn_down_blocks[i](noisy_images, cond)

        for i, layer in enumerate(self.decoder_blocks):
            noisy_images = layer(noisy_images)

            # Apply cross-attention for AttnUpBlock2D
            if isinstance(layer, CrossAttention) and i < len(self.attn_up_blocks):
                noisy_images = self.attn_up_blocks[i](noisy_images, cond)
    
        return noisy_images

def main():
    image_size = 224
    in_channels = 1
    condition_channels = 365
    batch_size = 1

    # Instantiate Encoder
    encoder = Encoder(
        model_weights=None,  # Provide the path if weights are required
        depth_model=True,
        constant_depth=None,
        device="cuda"
    ).to("cuda")

    # Instantiate Conditional UNet
    model = ConditionalUNet2DModel(
        sample_size=image_size,
        in_channels=in_channels,
        out_channels=in_channels,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        condition_channels=condition_channels,
    ).to("cuda")

    print(f"The model has {model.num_parameters():,} parameters")

    # Generate mock data
    rgb_images = torch.randn(batch_size, 3, image_size, image_size).to("cuda")
    depth_maps = torch.randn(batch_size, 1, image_size, image_size).to("cuda")
    timesteps = torch.randint(0, 1000, (batch_size,), device="cuda")
    noisy_images = torch.randn(batch_size, in_channels, 512, 512).to("cuda")

    # Forward pass through the encoder to obtain conditioning features
    conditions, _ = encoder(rgb_images, depth_maps)

    # Forward pass through Conditional UNet
    output = model(noisy_images, timesteps, conditions)

    # Verify output shape matches input shape
    assert output['sample'].shape == noisy_images.shape, f"Output shape {output.shape} does not match input shape {noisy_images.shape}"

    print(f"Test passed! Output shape: {output['sample'].shape}")

if __name__ == "__main__":
    main()