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
from diffusers.optimization import get_cosine_schedule_with_warmup
from scripts.model import ConditionalDDPM
from scripts.dataset import RIRDDMDataset
from scripts.stft import STFT
import numpy as np

LAMBDAS = [1, 0, 1e-2, 1e-2]  # LAMBDA multipliers for different losses
ADAM_BETA = (0.0, 0.99)
ADAM_EPS = 1e-8
NUM_TRAIN_TIMESTEPS = 30

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

    stft = STFT()
    writer = None
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.version))

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss_total = 0

        progress_bar = tqdm(train_loader, disable=not accelerator.is_main_process, desc=f"Epoch {epoch}/{args.epochs}")
        for B_spec, text_embedding, image_embedding, _ in progress_bar:
            B_spec = B_spec.to(device)
            text_embedding = text_embedding.to(device)
            image_embedding = image_embedding.to(device)

            noise = torch.randn_like(B_spec).to(device)
            bsz = B_spec.shape[0]
            indices = torch.randint(0, model.scheduler.config.num_train_timesteps, (bsz,))
            timesteps = model.scheduler.timesteps[indices].to(device)

            noisy_spectrogram = model.scheduler.add_noise(B_spec, noise, timesteps)
            model_input = model.scheduler.precondition_inputs(noisy_spectrogram, timesteps)
            predicted_noise = model(model_input, timesteps, text_embedding, image_embedding)
            denoised_sample = model.scheduler.precondition_outputs(noisy_spectrogram, predicted_noise, timesteps)

            loss_1 = criterion(noise, predicted_noise)
            loss = LAMBDAS[0] * loss_1

            # Backward pass
            with accelerator.accumulate(model):
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_loss_total += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        # Logging
        if accelerator.is_main_process:
            writer.add_scalar("Train/Total Loss", train_loss_total / len(train_loader), epoch)

        # Save checkpoint every 20% of epochs
        if (epoch + 1) % (args.epochs // 5) == 0 or epoch == args.epochs - 1:
            save_checkpoint(epoch)

        # Validation
        model.eval()
        val_loss_total = 0
        with torch.no_grad():
            for B_spec, text_embedding, image_embedding, _ in tqdm(val_loader, disable=not accelerator.is_main_process, desc="Validation"):
                B_spec = B_spec.to(device)
                text_embedding = text_embedding.to(device)
                image_embedding = image_embedding.to(device)

                noise = torch.randn_like(B_spec).to(device)
                indices = torch.randint(0, model.scheduler.config.num_train_timesteps, (B_spec.size(0),))
                timesteps = model.scheduler.timesteps[indices].to(device)

                noisy_spectrogram = model.scheduler.add_noise(B_spec, noise, timesteps)
                model_input = model.scheduler.precondition_inputs(noisy_spectrogram, timesteps)
                predicted_noise = model(model_input, timesteps, text_embedding, image_embedding)
                denoised_sample = model.scheduler.precondition_outputs(noisy_spectrogram, predicted_noise, timesteps)

                loss_1 = criterion(noise, predicted_noise)
                val_loss_total += loss_1.item()

        if accelerator.is_main_process:
            writer.add_scalar("Validation/Total Loss", val_loss_total / len(val_loader), epoch)

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

    model = ConditionalDDPM(
        noise_channels=1, embedding_dim=512, image_size=512, num_train_timesteps=NUM_TRAIN_TIMESTEPS
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=ADAM_BETA, eps=ADAM_EPS)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=len(train_loader) * args.epochs)

    criterion = nn.MSELoss()
    lpips_loss = lpips.LPIPS(net="vgg").to(device)

    # Load checkpoint if specified
    start_epoch = 0
    best_val_loss = float("inf")
    if args.from_pretrained:
        checkpoint_path = os.path.join(args.checkpoint_dir, args.version, args.from_pretrained)
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint["best_val_loss"]
        else:
            print(f"Checkpoint {checkpoint_path} not found. Starting from scratch.")

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
        start_epoch=start_epoch,
        best_val_loss=best_val_loss,
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
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory to save TensorBoard logs.")
    parser.add_argument("--version", type=str, default="trial_08", help="Experiment version.")
    parser.add_argument("--from_pretrained", type=str, default=None, help="Checkpoint filename to resume training.")
    args = parser.parse_args()

    main(args)
