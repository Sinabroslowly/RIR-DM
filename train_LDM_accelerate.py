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
from diffusers import EDMDPMSolverMultistepScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from scripts.model import LDT, FeatureMapGenerator
from scripts.dataset import RIRDDMDataset
from scripts.stft import STFT
#from scripts.util import compare_t60, compare_t60_octave_bandwise, weighted_t60_err
import numpy as np

LAMBDAS = [1, 0, 1e-2, 1e-2]  # LAMBDA multipliers for different losses
ADAM_BETA = (0.0, 0.99)
ADAM_EPS = 1e-8
NUM_TRAIN_TIMESTEPS = 999
NUM_INFERENCE_TIMESTEPS = 30

def train_model(model, optimizer, criterion, scheduler, lpips_loss, train_loader, val_loader, device, start_epoch, best_val_loss, args, accelerator):
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
        checkpoint_name = f"checkpoint_epoch_{epoch}.pth"
        checkpoint_path = os.path.join(args.checkpoint_dir, args.version, checkpoint_name)
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = os.path.join(args.checkpoint_dir, args.version, "best_checkpoint.pth")
            torch.save(checkpoint, best_path)

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = model.scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = model.scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)

        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    stft = STFT()
    feature_extractor = FeatureMapGenerator()
    writer = None
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.version))

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss_total = 0
        train_loss_1 = 0
        train_loss_2 = 0
        train_loss_3 = 0
        train_loss_4 = 0

        ddpm_scheduler = EDMDPMSolverMultistepScheduler(sigma_min=0.002, 
                                                        sigma_max=80.0, 
                                                        sigma_data=0.5,
                                                        sigma_schedule='karras',
                                                        solver_order=2,
                                                        prediction_type='epsilon',
                                                        num_train_timesteps=NUM_TRAIN_TIMESTEPS) # Noise scheduler
        ddpm_scheduler.set_timesteps(num_inference_steps = NUM_INFERENCE_TIMESTEPS)

        progress_bar = tqdm(train_loader, disable=not accelerator.is_main_process, desc=f"Epoch {epoch}/{args.epochs}")
        for latent_spec_matrix, text_embedding, image_embedding, _ in progress_bar:
            latent_spec_matrix = latent_spec_matrix.to(device)
            text_embedding = text_embedding.to(device)
            image_embedding = image_embedding.to(device)

            cross_modal_embedding = feature_extractor(text_embedding, image_embedding)

            noise = torch.randn_like(latent_spec_matrix).to(device)
            bsz = latent_spec_matrix.shape[0]
            indices = torch.randint(0, model.scheduler.config.num_train_timesteps, (bsz,))
            timesteps = model.scheduler.timesteps[indices].to(device)

            noisy_latent_matrix = model.scheduler.add_noise(latent_spec_matrix, noise, timesteps)
            sigmas = get_sigmas(timesteps, len(noisy_latent_matrix.shape), noisy_latent_matrix.dtype)
            sigma_noisy_latent_matrix = model.scheduler.precondition_inputs(noisy_latent_matrix, sigmas)
            predicted_noise = model(sigma_noisy_latent_matrix, cross_modal_embedding, timesteps)
            denoised_sample = model.scheduler.precondition_outputs(noisy_latent_matrix, predicted_noise, sigmas)

            # Loss Calculations
            loss_1 = criterion(noise, predicted_noise)  # Noise prediction loss


            loss = LAMBDAS[0] * loss_1 # Only use the noise reconstruction loss.

            with accelerator.accumulate(model):
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_loss_total += loss.item()
            train_loss_1 += loss_1.item()

            progress_bar.set_postfix(loss=loss.item())
        # Logging
        if accelerator.is_main_process:
            writer.add_scalar("Train/Total Loss", train_loss_total / len(train_loader), epoch)
            writer.add_scalar("Train/Noise Loss", train_loss_1 / len(train_loader), epoch)
            #writer.add_scalar("Train/Reconstruction Loss", train_loss_2 / len(train_loader), epoch)
            #writer.add_scalar("Train/Octave Band Loss", train_loss_3 / len(train_loader), epoch)
            #writer.add_scalar("Train/T60 PRA Loss", train_loss_4 / len(train_loader), epoch)

        # Save checkpoint every 10% of epochs
        if (epoch + 1) % (args.epochs // 10) == 0 or epoch == args.epochs - 1:
            save_checkpoint(epoch)

        # Validation
        model.eval()
        val_loss_total = 0
        val_loss_1 = 0
        # val_loss_2 = 0
        # val_loss_3 = 0
        # val_loss_4 = 0
        with torch.no_grad():
            for latent_spec_matrix, text_embedding, image_embedding, _ in tqdm(val_loader, disable=not accelerator.is_main_process, desc="Validation"):
                latent_spec_matrix = latent_spec_matrix.to(device)
                text_embedding = text_embedding.to(device)
                image_embedding = image_embedding.to(device)

                cross_modal_embedding = feature_extractor(text_embedding, image_embedding)

                noise = torch.randn_like(latent_spec_matrix).to(device)
                bsz = latent_spec_matrix.shape[0]
                indices = torch.randint(0, model.scheduler.config.num_train_timesteps, (bsz,))
                timesteps = model.scheduler.timesteps[indices].to(device)

                noisy_latent_matrix = model.scheduler.add_noise(latent_spec_matrix, noise, timesteps)
                sigmas = get_sigmas(timesteps, len(noisy_latent_matrix.shape), noisy_latent_matrix.dtype)
                sigma_noisy_latent_matrix = model.scheduler.precondition_inputs(noisy_latent_matrix, sigmas)
                predicted_noise = model(sigma_noisy_latent_matrix, cross_modal_embedding, timesteps)
                denoised_sample = model.scheduler.precondition_outputs(noisy_latent_matrix, predicted_noise, sigmas)

                loss_1 = criterion(noise, predicted_noise)
                # loss_2 = criterion(B_spec, denoised_sample)
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
                # val_loss_2 += loss_2.item()
                # val_loss_3 += loss_3.item()
                # val_loss_4 += loss_4

                val_loss_total += LAMBDAS[0] * loss_1.item()
                                  #  LAMBDAS[1] * loss_2.item() +
                                  #  LAMBDAS[2] * loss_3.item() +
                                  #  LAMBDAS[3] * loss_4)

            latent_noise = torch.randn(latent_spec_matrix.shape, device=device) * model.scheduler.init_noise_sigma
            intermediate_noise = []

            for i, t in enumerate(ddpm_scheduler.timesteps):
                if (i + 1) % (len(ddpm_scheduler.timesteps)/5) == 0:
                    intermediate_noise.append(latent_noise.cpu().squeeze().detach())
                model_input = ddpm_scheduler.scale_model_input(latent_noise, t)
                predicted_noise = model(model_input, t, text_embedding, image_embedding)
                latent_noise = ddpm_scheduler.step(predicted_noise, t, latent_noise).prev_sample
            combined_intermediate_noise = torch.clamp(torch.cat(intermediate_noise, dim=-1), min=-0.8, max=0.8)
            combined_intermediate_noise = torch.cat([combined_intermediate_noise[:,0,:,:], combined_intermediate_noise[:,-1,:,:]], dim=-2) # Adding the first and last latent representation of the intermediate noise.

            if torch.isnan(combined_intermediate_noise).any():
                print(f"Warning: Image contains NaN values")
            if torch.isinf(combined_intermediate_noise).any():
                print(f"Warning: Image contains inf values.")

        if accelerator.is_main_process:
            writer.add_scalar("Validation/Total Loss", val_loss_total / len(val_loader), epoch)
            writer.add_scalar("Validation/Noise Loss", val_loss_1 / len(val_loader), epoch)
            # writer.add_scalar("Validation/Reconstruction Loss", val_loss_2 / len(val_loader), epoch)
            # writer.add_scalar("Validation/Octave Band Loss", val_loss_3 / len(val_loader), epoch)
            # writer.add_scalar("Validation/T60 PRA Loss", val_loss_4 / len(val_loader), epoch)
            if not (torch.isnan(combined_intermediate_noise).any() or torch.isinf(combined_intermediate_noise).any()):
              writer.add_image("Spectrogram/Intermediate Denoising", combined_intermediate_noise, epoch)

    if writer:
        writer.close()

def main(args):
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device

    os.makedirs(os.path.join(args.checkpoint_dir, args.version), exist_ok=True)

    train_dataset = RIRDDMDataset(dataroot=args.data_dir, device=device, phase="train")
    val_dataset = RIRDDMDataset(dataroot=args.data_dir, device=device, phase="val")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = LDT(
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
        num_train_timesteps=NUM_TRAIN_TIMESTEPS,
    )

    model.scheduler.set_timesteps(num_inference_steps = NUM_TRAIN_TIMESTEPS)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=ADAM_BETA, eps=ADAM_EPS)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=len(train_loader) * args.epochs)
    criterion = nn.MSELoss()
    #lpips_loss = lpips.LPIPS(net="vgg").to(device)

    # Load checkpoint if available
    start_epoch = 0
    best_val_loss = float("inf")
    if args.from_pretrained:
        checkpoint_path = os.path.join(args.checkpoint_dir, args.version, args.from_pretrained)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]

    # Prepare for training
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    train_model(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        lpips_loss=lpips_loss,
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
    parser.add_argument("--data_dir", type=str, default="./datasets_subset_complete", help="Path to the dataset.")
    #parser.add_argument("--data_dir", type=str, default="./datasets", help="Path to the dataset.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=100, help="Total number of epochs.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--t60_ratio", type=float, default=1.0, help="Ratio between broadband and octave-band t60 loss.")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory to save TensorBoard logs.")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--version", type=str, default="trial_10", help="Experiment version.")
    parser.add_argument("--from_pretrained", type=str, default=None, help="Path to a checkpoint to resume training.")
    args = parser.parse_args()

    main(args)
