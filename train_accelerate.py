from accelerate import Accelerator
import os
import torch
import torch.nn as nn
import torch.optim as optim
import lpips
import pyroomacoustics
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm
from argparse import ArgumentParser
from diffusers import EDMEulerScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from scripts.model import ConditionalDDPM
from scripts.dataset import RIRDDMDataset
from scripts.util import compare_t60, estimate_t60, compare_t60_octave_bandwise, weighted_t60_err
from scripts.stft import STFT
import numpy as np

LAMBDAS = [1, 0, 1e-2, 1e-2]  # LAMBDA multipliers for different losses
ADAM_BETA = (0.0, 0.99)
ADAM_EPS = 1e-8
NUM_TRAIN_TIMESTEPS = 30

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def octave_band_t60_error_loss(fake_spec, spec, device, t60_ratio=0.5):
    t60_err_fs = torch.Tensor([compare_t60(torch.pow(10, a).sum(-2).squeeze(), torch.pow(10, b).sum(-2).squeeze()) for a, b in zip(spec, fake_spec)]).to(device).mean()
    t60_errs = torch.Tensor([compare_t60_octave_bandwise(a, b) for a, b in zip(spec, fake_spec)]).to(device)
    t60_err_bs = weighted_t60_err(t60_errs)
    return ((1 - t60_ratio) * t60_err_fs + t60_ratio * t60_err_bs)

def train_model(model, optimizer, criterion, scheduler, lpips_loss, train_loader, val_loader, device, start_epoch, best_val_loss, args, accelerator):
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
    global_step = 0

    writer = None
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.version))

    for epoch in range(start_epoch + 1, args.epochs):
        model.train()
        train_loss_total = 0
        train_loss_1 = 0
        train_loss_2 = 0
        train_loss_3 = 0
        train_loss_4 = 0

        progress_bar = tqdm(train_loader, disable=not accelerator.is_main_process, desc=f"Epoch {epoch}/{args.epochs}")
        for B_spec, text_embedding, image_embedding, _ in progress_bar:
            B_spec = B_spec.to(device)
            text_embedding = text_embedding.to(device)
            image_embedding = image_embedding.to(device)

            noise = torch.randn_like(B_spec).to(device)
            bsz = B_spec.shape[0]
            indices = torch.randint(0, model.scheduler.config.num_train_timesteps, (bsz,))
            timesteps = model.scheduler.timesteps[indices].to(device)

            # Add noise and predict
            noisy_spectrogram = model.scheduler.add_noise(B_spec, noise, timesteps)
            sigmas = get_sigmas(timesteps, len(noisy_spectrogram.shape), noisy_spectrogram.dtype)
            sigma_noisy_spectrogram = model.scheduler.precondition_inputs(noisy_spectrogram, sigmas)
            predicted_noise = model(sigma_noisy_spectrogram, timesteps, text_embedding, image_embedding)
            denoised_sample = model.scheduler.precondition_outputs(noisy_spectrogram, predicted_noise, sigmas)

            # Loss calculations
            loss_1 = criterion(noise, predicted_noise)  # Noise prediction loss
            loss_2 = criterion(B_spec, denoised_sample)  # Reconstruction loss
            loss_3 = octave_band_t60_error_loss(B_spec, denoised_sample, device, args.t60_ratio) * ((2*((epoch / args.epochs)-0.5))**2)
            
            y_r = [stft.inverse(s.squeeze()) for s in B_spec]
            y_f = [stft.inverse(s.squeeze()) for s in denoised_sample]
            try:
                t60_r = [pyroomacoustics.experimental.rt60.measure_rt60(y, 22050) for y in y_r]
                t60_f = [pyroomacoustics.experimental.rt60.measure_rt60(y, 22050) for y in y_f]
                loss_4 = np.mean([(b - a) / a for a, b in zip(t60_r, t60_f)])
            except:
                loss_4 = 0

            loss = (LAMBDAS[0] * loss_1 + 
                    LAMBDAS[1] * loss_2 + 
                    LAMBDAS[2] * loss_3 + 
                    LAMBDAS[3] * loss_4)

            # Gradient accumulation and optimizer step
            with accelerator.accumulate(model):
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_loss_total += loss.item()
            train_loss_1 += loss_1.item()
            train_loss_2 += loss_2.item()
            train_loss_3 += loss_3.item()
            train_loss_4 += loss_4

            progress_bar.set_postfix(loss=loss.item())

        # Logging
        if accelerator.is_main_process:
            writer.add_scalar("Train/Total Loss", train_loss_total / len(train_loader), epoch)
            writer.add_scalar("Train/Noise Loss", train_loss_1 / len(train_loader), epoch)
            writer.add_scalar("Train/Reconstruction Loss", train_loss_2 / len(train_loader), epoch)
            writer.add_scalar("Train/Octave Band Loss", train_loss_3 / len(train_loader), epoch)
            writer.add_scalar("Train/T60 PRA Loss", train_loss_4 / len(train_loader), epoch)

        # Validation
        model.eval()
        val_loss_1 = 0
        val_loss_2 = 0
        val_loss_3 = 0
        val_loss_4 = 0
        val_images_flag = False

        with torch.no_grad():
            for B_spec, text_embedding, image_embedding, _ in tqdm(val_loader, disable=not accelerator.is_main_process, desc="Validation"):
                B_spec = B_spec.to(device)
                text_embedding = text_embedding.to(device)
                image_embedding = image_embedding.to(device)

                noise = torch.randn_like(B_spec).to(device)
                bsz = B_spec.shape[0]
                indices = torch.randint(0, model.scheduler.config.num_train_timesteps, (bsz,))
                timesteps = model.scheduler.timesteps[indices].to(device)

                noisy_spectrogram = model.scheduler.add_noise(B_spec, noise, timesteps)
                sigmas = get_sigmas(timesteps, len(noisy_spectrogram.shape), noisy_spectrogram.dtype)
                sigma_noisy_spectrogram = model.scheduler.precondition_inputs(noisy_spectrogram, sigmas)
                predicted_noise = model(sigma_noisy_spectrogram, timesteps, text_embedding, image_embedding)
                denoised_sample = model.scheduler.precondition_outputs(noisy_spectrogram, predicted_noise, sigmas)

                loss_1 = criterion(noise, predicted_noise)
                loss_2 = criterion(B_spec, denoised_sample)
                loss_3 = octave_band_t60_error_loss(B_spec, denoised_sample, device, args.t60_ratio)
                
                y_r = [stft.inverse(s.squeeze()) for s in B_spec]
                y_f = [stft.inverse(s.squeeze()) for s in denoised_sample]
                try:
                    t60_r = [pyroomacoustics.experimental.rt60.measure_rt60(y, 22050) for y in y_r]
                    t60_f = [pyroomacoustics.experimental.rt60.measure_rt60(y, 22050) for y in y_f]
                    loss_4 = np.mean([(b - a) / a for a, b in zip(t60_r, t60_f)])
                except:
                    loss_4 = 0

                val_loss_1 += loss_1.item()
                val_loss_2 += loss_2.item()
                val_loss_3 += loss_3.item()
                val_loss_4 += loss_4

                if not val_images_flag:
                    reconstructed_spectrogram = model.scheduler.precondition_outputs(noisy_spectrogram, predicted_noise, sigmas)
                    val_images_gt = B_spec
                    val_images_fake = reconstructed_spectrogram
                    val_images_flag = True

        if accelerator.is_main_process:
            writer.add_scalar("Validation/Noise Loss", val_loss_1 / len(val_loader), epoch)
            writer.add_scalar("Validation/Reconstruction Loss", val_loss_2 / len(val_loader), epoch)
            writer.add_scalar("Validation/Octave Band Loss", val_loss_3 / len(val_loader), epoch)
            writer.add_scalar("Validation/T60 PRA Loss", val_loss_4 / len(val_loader), epoch)

            if val_images_flag:
                gt_spec, gen_spec = val_images_gt, val_images_fake
                writer.add_image("Spectrogram/Ground Truth", make_grid(gt_spec.cpu(), normalize=True), epoch)
                writer.add_image("Spectrogram/Generated", make_grid(gen_spec.cpu(), normalize=True), epoch)
            

    if writer:
        writer.close()

def main(args):
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device

    train_dataset = RIRDDMDataset(dataroot=args.data_dir, device=device, phase="train")
    val_dataset = RIRDDMDataset(dataroot=args.data_dir, device=device, phase="val")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = ConditionalDDPM(
        noise_channels=1, embedding_dim=512, image_size=512, num_train_timesteps=NUM_TRAIN_TIMESTEPS
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=ADAM_BETA, eps=ADAM_EPS)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=len(train_loader) * args.epochs)

    criterion = nn.MSELoss()
    lpips_loss = lpips.LPIPS(net="vgg").to(device)

    # Prepare with Accelerator
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
        start_epoch=0,
        best_val_loss=float("inf"),
        args=args,
        accelerator=accelerator,
    )

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./datasets", help="Path to the dataset.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=50, help="Total number of epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--t60_ratio", type=float, default=0.5, help="Ratio between broadband and octave-band t60 loss.")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory to save TensorBoard logs.")
    parser.add_argument("--version", type=str, default="trial_08", help="Experiment version.")
    args = parser.parse_args()

    main(args)
