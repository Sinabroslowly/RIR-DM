from accelerate import Accelerator
import os
import torch
import torch.nn as nn
import torch.optim as optim
#import lpips
#import pyroomacoustics
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm
from argparse import ArgumentParser
from diffusers import DDIMScheduler, VQModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from scripts.model import LDT, FeatureMapGenerator
from scripts.dataset import RIRDDMDataset
from scripts.stft import STFT
#from scripts.util import compare_t60, compare_t60_octave_bandwise, weighted_t60_err
import numpy as np

LAMBDAS = [1, 1, 1e-2, 1e-2]  # LAMBDA multipliers for different losses
ADAM_BETA = (0.0, 0.99)
ADAM_EPS = 1e-8
NUM_TRAIN_TIMESTEPS = 1000
NUM_INFERENCE_TIMESTEPS = 200

def load_vqvae_decoder(checkpoint_path, device):
        """Load the VQ-VAE model and extract the decoder."""
        model = VQModel(
            in_channels=1,
            out_channels=1,
            down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
            up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
            block_out_channels=(32, 64, 128, 256),
            layers_per_block=3, # 1 for trial_01 with 100 epoch, 3 for trial_02 with 600 epoch.
            act_fn="silu",
            sample_size=512,
            latent_channels=8,
            num_vq_embeddings=256,
            scaling_factor=0.18215,
        )
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model_state_dict = checkpoint["model_state"]
        #opti_state_dict = checkpoint["optimizer_state"]
        if any(key.startswith("module.") for key in model_state_dict.keys()):
            model_state_dict = {key[len("module."):]: value for key, value in model_state_dict.items()}
        model.load_state_dict(model_state_dict)

        decoder = model.decoder
        del model

        for param in decoder.parameters():
            param.requires_grad = False
        decoder.eval()
        
        return decoder.to(device)

def train_model(model, vqvae_decoder, optimizer, criterion, scheduler, train_loader, val_loader, device, start_epoch, best_val_loss, args, accelerator):
    def save_checkpoint(epoch, is_best=False):
        """
        Save the model checkpoint at specified intervals.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state": accelerator.unwrap_model(model).state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
        }
        if not is_best:
            checkpoint_name = f"checkpoint_epoch_{epoch}.pth"
            checkpoint_path = os.path.join(args.checkpoints_dir, args.version, checkpoint_name)
            torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = os.path.join(args.checkpoints_dir, args.version, "best_validation.pth")
            torch.save(checkpoint, best_path)

    torch.manual_seed(0)
    #stft = STFT()
    feature_extractor = FeatureMapGenerator()
    writer = None
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.version))

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss_total = 0
        train_loss_1 = 0
        #train_loss_2 = 0
        # train_loss_3 = 0
        # train_loss_4 = 0

        ddim_scheduler = DDIMScheduler(beta_start = 0.0001, 
                                                        beta_end=0.02,
                                                        beta_schedule='linear',
                                                        clip_sample_range=0.8,
                                                        prediction_type='epsilon',
                                                        num_train_timesteps=NUM_TRAIN_TIMESTEPS) # Noise scheduler
        ddim_scheduler.set_timesteps(num_inference_steps = NUM_INFERENCE_TIMESTEPS)

        progress_bar = tqdm(train_loader, disable=not accelerator.is_main_process, desc=f"Epoch {epoch}/{args.epochs}")
        for latent_spec_matrix, text_embedding, image_embedding, _ in progress_bar:
            latent_spec_matrix = latent_spec_matrix.to(device)
            text_embedding = text_embedding.to(device)
            image_embedding = image_embedding.to(device)

            cross_modal_embedding = feature_extractor(text_embedding, image_embedding)


            noise = torch.rand(latent_spec_matrix.shape).to(device)
            bsz = latent_spec_matrix.shape[0]
            timesteps = torch.randint(0, model.module.scheduler.config.num_train_timesteps, (bsz,), device=device, dtype=torch.int64)

            noisy_latent_matrix = model.module.scheduler.add_noise(latent_spec_matrix, noise, timesteps)
            predicted_noise = model.module(noisy_latent_matrix, 
                cross_modal_embedding, 
                timesteps, 
            )
            #predicted_latent = model.module.scheduler.step(predicted_noise, timesteps, noisy_latent_matrix)

            #gt_spec, gen_spec = vqvae_decoder(latent_spec_matrix), vqvae_decoder(predicted_latent)
            loss_1 = criterion(noise, predicted_noise)
            #loss_2 = criterion(gt_spec, gen_spec)

            loss = LAMBDAS[0] * loss_1# + LAMBDAS[1] * loss_2 # Only use the noise reconstruction loss.

            with accelerator.accumulate(model):
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_loss_total += loss.item()
            train_loss_1 += loss_1.item()
            #train_loss_2 += loss_2.item()

            progress_bar.set_postfix(loss=loss.item())
        # Logging
        if accelerator.is_main_process:
            writer.add_scalar("Train/Total Loss", train_loss_total / len(train_loader), epoch)
            writer.add_scalar("Train/Noise Loss", train_loss_1 / len(train_loader), epoch)
            #writer.add_scalar("Train/Reconstruction Loss", train_loss_2 / len(train_loader), epoch)
            #writer.add_scalar("Train/Octave Band Loss", train_loss_3 / len(train_loader), epoch)
            #writer.add_scalar("Train/T60 PRA Loss", train_loss_4 / len(train_loader), epoch)


        # Validation
        model.eval()
        val_loss_total = 0
        val_loss_1 = 0
        val_loss_2 = 0
        # val_loss_3 = 0
        # val_loss_4 = 0
        with torch.no_grad():
            progress_bar_val = tqdm(val_loader, disable=not accelerator.is_main_process, desc=f"Epoch {epoch}/{args.epochs}")
            for latent_spec_matrix, text_embedding, image_embedding, _ in progress_bar_val:
                latent_spec_matrix = latent_spec_matrix.to(device)
                text_embedding = text_embedding.to(device)
                image_embedding = image_embedding.to(device)

                cross_modal_embedding = feature_extractor(text_embedding, image_embedding)

                noise = torch.rand(latent_spec_matrix.shape).to(device)
                bsz = latent_spec_matrix.shape[0]
                timesteps = torch.randint(0, model.module.scheduler.config.num_train_timesteps, (bsz,), device=device)

                noisy_latent_matrix = model.module.scheduler.add_noise(latent_spec_matrix, noise, timesteps)
                predicted_noise = model.module(noisy_latent_matrix, 
                cross_modal_embedding, 
                timesteps, 
                )
                #print(f"Shape of predicted_noise: {predicted_noise.shape}")
                #predicted_latent = model.module.scheduler.step(predicted_noise, timesteps, noisy_latent_matrix)

                #gt_spec, gen_spec = vqvae_decoder(latent_spec_matrix), vqvae_decoder(predicted_latent)

                loss_1 = criterion(noise, predicted_noise)
                #loss_2 = criterion(gt_spec, gen_spec)
                # loss_3 = octave_band_t60_error_loss(B_spec, denoised_sample, device, args.t60_ratio)

                # y_r = [stft.inverse(s.squeeze()) for s in B_spec]
                # y_f = [stft.inverse(s.squeeze()) for s in denoised_sample]
                # try:
                #     t60_r = [pyroomacoustics.experimental.rt60.measure_rt60(y, 22050) for y in y_r]
                #     t60_f = [pyroomacoustics.experimental.rt60.measure_rt60(y, 22050) for y in y_f]
                #     loss_4 = np.mean([(b - a) / a for a, b in zip(t60_r, t60_f)])
                # except:
                #     loss_4 = 0

                val_loss_1 += loss_1.item()
                #val_loss_2 += loss_2.item()
                # val_loss_3 += loss_3.item()
                # val_loss_4 += loss_4

                val_loss_total += LAMBDAS[0] * loss_1.item()# + LAMBDAS[1] * loss_2.item()
                                  #  LAMBDAS[2] * loss_3.item() +
                                  #  LAMBDAS[3] * loss_4)

                progress_bar.set_postfix(loss=val_loss_total)

            
            if accelerator.is_main_process:
                latent_noise = torch.randn(latent_spec_matrix.shape, device=device)
                intermediate_noise = []
                for i, t in enumerate(ddim_scheduler.timesteps):
                    #print(f"timestep t: {t}")
                    if i == 0 or i % (len(ddim_scheduler.timesteps)/5) == 0 or (i+1) == len(ddim_scheduler.timesteps):
                        intermediate_spec = vqvae_decoder(latent_noise)
                        intermediate_noise.append(intermediate_spec.cpu().squeeze().detach())
                    predicted_noise = model.module(latent_noise, 
                    cross_modal_embedding, 
                    t.unsqueeze(0).to(device), 
                    )
                    latent_noise = ddim_scheduler.step(predicted_noise, t, latent_noise).prev_sample
                #combined_intermediate_noise = torch.clamp(torch.cat(intermediate_noise, dim=-1), min=-0.8, max=0.8)
                combined_intermediate_noise = torch.cat(intermediate_noise, dim=-1)
                #combined_intermediate_noise = torch.cat([combined_intermediate_noise[:,0,:,:], combined_intermediate_noise[:,-1,:,:]], dim=-2) # Adding the first and last latent representation of the intermediate noise.

                if torch.isnan(combined_intermediate_noise).any():
                    print(f"Warning: Image contains NaN values")
                if torch.isinf(combined_intermediate_noise).any():
                    print(f"Warning: Image contains inf values.")

        if accelerator.is_main_process:
            writer.add_scalar("Validation/Total Loss", val_loss_total / len(val_loader), epoch)
            writer.add_scalar("Validation/Noise Loss", val_loss_1 / len(val_loader), epoch)
            #writer.add_scalar("Validation/Reconstruction Loss", val_loss_2 / len(val_loader), epoch)
            # writer.add_scalar("Validation/Octave Band Loss", val_loss_3 / len(val_loader), epoch)
            # writer.add_scalar("Validation/T60 PRA Loss", val_loss_4 / len(val_loader), epoch)
            #recon_grid = make_grid(predicted_latent[0].unsqueeze(1), nrow=4, normalize=True, scale_each=True)
            recon_grid = make_grid(latent_noise[0].unsqueeze(1), nrow=4, normalize=True, scale_each=True)
            gt_grid = make_grid(latent_spec_matrix[0].unsqueeze(1), nrow=4, normalize=True, scale_each=True)
            writer.add_image("Spectrogram/Latent Space - Reconstructed", recon_grid, epoch)
            writer.add_image("Spectrogram/Latent Space - Ground Truth", gt_grid, epoch)
            if not (torch.isnan(combined_intermediate_noise).any() or torch.isinf(combined_intermediate_noise).any()):
              writer.add_image("Spectrogram/Intermediate Denoising", combined_intermediate_noise[0].unsqueeze(0), epoch)

            # Save checkpoint every 5% of epochs
            if (epoch + 1) % (args.epochs // 20) == 0 or epoch == args.epochs - 1:
                save_checkpoint(epoch, is_best=False)
            if val_loss_total < best_val_loss:
                save_checkpoint(epoch, is_best=True)
                best_val_loss = val_loss_total



    if writer:
        writer.close()

def main(args):
    accelerator = Accelerator(mixed_precision="no")
    #accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device

    os.makedirs(os.path.join(args.checkpoints_dir, args.version), exist_ok=True)

    train_dataset = RIRDDMDataset(dataroot=args.data_dir, device=device, phase="train")
    val_dataset = RIRDDMDataset(dataroot=args.data_dir, device=device, phase="val")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = LDT(
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
        num_train_timesteps=NUM_TRAIN_TIMESTEPS,
    )
    if accelerator.is_main_process:
        print(f"Total {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters in LDM model.")

    model.scheduler.set_timesteps(num_inference_steps = NUM_TRAIN_TIMESTEPS)

    # Explicitly disable xformers attention for stable diffusion (Preventing NAN Loss)
    # if hasattr(model, "enable_xformers_memory_efficient_attention"):
    #     model.enable_xformers_memory_efficient_attention(False)

    vqvae_decoder = load_vqvae_decoder(args.vqvae_checkpoint, device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=ADAM_BETA, eps=ADAM_EPS)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=len(train_loader) * args.epochs)
    criterion = nn.MSELoss()
    #lpips_loss = lpips.LPIPS(net="vgg").to(device)

    # Load checkpoint if available
    start_epoch = 1
    best_val_loss = float("inf")
    if args.from_pretrained:
        checkpoint = torch.load(os.path.join(args.checkpoints_dir, args.version, args.from_pretrained), map_location=device, weights_only=True)
        model_state_dict = checkpoint["model_state"]
        #opti_state_dict = checkpoint["optimizer_state"]
        if any(key.startswith("module.") for key in model_state_dict.keys()):
            model_state_dict = {key[len("module."):]: value for key, value in model_state_dict.items()}
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        #print(f"Loaded checkpoint from: {args.from_pretrained}")

    # Prepare for training
    model, vqvae_decoder, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, vqvae_decoder, optimizer, train_loader, val_loader, scheduler
    )

    accelerator.clip_grad_norm_(model.parameters(), max_norm=0.5) # Suggestion to prevent NaN.

    train_model(
        model=model,
        vqvae_decoder=vqvae_decoder,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        start_epoch=start_epoch,
        best_val_loss=best_val_loss,
        args=args,
        accelerator=accelerator,
    )

if __name__ == "__main__":
    parser = ArgumentParser()
    #parser.add_argument("--data_dir", type=str, default="./datasets_subset_complete", help="Path to the dataset.")
    parser.add_argument("--data_dir", type=str, default="./datasets", help="Path to the dataset.")
    parser.add_argument("--vqvae_checkpoint", type=str, default="./VQVAE/trial_02_epoch_600_checkpoint.pth", help="Path to pretrained VQ-VAE checkpoint.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=500, help="Total number of epochs.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--t60_ratio", type=float, default=1.0, help="Ratio between broadband and octave-band t60 loss.")
    parser.add_argument("--p_cond", type=float, default=0.2, help="The probability to apply unconditioned denoising for CFG")
    parser.add_argument("--cfg_weight", type=float, default=2.0, help="The value of classifier-free-guidance parameter.")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory to save TensorBoard logs.")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--version", type=str, default="trial_14", help="Experiment version.")
    parser.add_argument("--from_pretrained", type=str, default="best_validation.pth", help="Path to a checkpoint to resume training.")
    #parser.add_argument("--from_pretrained", type=str, default=None, help="Path to a checkpoint to resume training.")
    args = parser.parse_args()

    main(args)
