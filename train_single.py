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
from scripts.model import ConditionalDDPM
from scripts.dataset import RIRDDMDataset
from scripts.util import compare_t60, estimate_t60, compare_t60_octave_bandwise, weighted_t60_err
from scripts.stft import STFT
import numpy as np

LAMBDAS = [1e+3, 1] # LAMBDA multiplication for spectrogram reconstruction L1 loss, t60 error loss and lpips loss.
ADAM_BETA = (0.0, 0.99)
ADAM_EPS = 1e-8
NUM_TRAIN_TIMESTEPS = 500

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def octave_band_t60_error_loss(fake_spec, spec, device, t60_ratio=0.5):
    t60_err_fs = torch.Tensor([compare_t60(torch.pow(10,a).sum(-2).squeeze(), torch.pow(10,b).sum(-2).squeeze()) for a, b in zip(spec, fake_spec)]).to(device).mean()
    t60_errs = torch.Tensor([compare_t60_octave_bandwise(a, b) for a, b in zip(spec, fake_spec)]).to(device)
    t60_err_bs = weighted_t60_err(t60_errs)
    return ((1-t60_ratio)*t60_err_fs + t60_ratio*t60_err_bs)

def train_model(model, optimizer, criterion, scheduler, lpips_loss, train_loader, val_loader, device, start_epoch, best_val_loss, args):

    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.version))
    # Training loop
    for epoch in range(start_epoch+1, args.epochs):
        model.train()
        train_loss = 0
        for B_spec, text_embedding, image_embedding, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            # Split E_embedding into text and image embeddings

            B_spec = B_spec.to(device)  # Target spectrogram
            text_embedding = text_embedding.to(device)
            image_embedding = image_embedding.to(device)

            # Scheduler timestep
            noise = torch.randn_like(B_spec).to(device)
            timesteps = torch.randint(0, model.scheduler.config.num_train_timesteps, (B_spec.size(0),), device=device)
            noisy_spectrogram = model.scheduler.add_noise(B_spec, noise, timesteps)

            # Forward pass
            optimizer.zero_grad()
            predicted_noise = model(noisy_spectrogram, timesteps, text_embedding, image_embedding)
            loss_1 = criterion(noise, predicted_noise) # Reconstruction Loss
            loss_2 = octave_band_t60_error_loss(predicted_noise, B_spec, device, args.t60_ratio)

            loss = LAMBDAS[0] * loss_1 + LAMBDAS[1] * loss_2
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            
        train_loss_tensor = torch.tensor(train_loss, device=device)
        # Aggregate losses across GPUs (simple sum)

        training_loss /= len(train_loader)
        writer.add_scalar("Loss/Train", training_loss, epoch)
        print(f"Epoch {epoch}, Train Loss: {training_loss}")

        # Validation loop
        model.eval()
        val_loss1 = 0
        val_loss2 = 0
        val_loss3 = 0

        val_images_flag = False

        with torch.no_grad():
            for B_spec, text_embedding, image_embedding, _ in tqdm(val_loader, desc="Validation", leave=False):
                # Split E_embedding into text and image embeddings

                B_spec = B_spec.to(device)
                text_embedding = text_embedding.to(device)
                image_embedding = image_embedding.to(device)

                noise = torch.randn_like(B_spec).to(device)
                timesteps = torch.randint(0, NUM_TRAIN_TIMESTEPS, (B_spec.size(0),), device=device)
                # Generate a fixed timestep tensor
                # timesteps = torch.full(
                #     (B_spec.size(0),),  # Same batch size as input
                #     fill_value=NUM_TRAIN_TIMESTEPS - 1,  # Fixed timestep (e.g., 999 if num_train_timesteps=1000)
                #     device=device,
                #     dtype=torch.long  # Ensure timesteps are long integers
                # )

                noisy_spectrogram = model.scheduler.add_noise(B_spec, noise, timesteps)

                predicted_noise = model(noisy_spectrogram, timesteps, text_embedding, image_embedding)
                loss_1 = criterion(noise, predicted_noise) # Reconstruction Loss
                loss_2 = octave_band_t60_error_loss(predicted_noise, B_spec, device, args.t60_ratio)

                with torch.no_grad():
                    loss_3 = lpips_loss(noise.repeat(1, 3, 1, 1), predicted_noise.repeat(1, 3, 1, 1)).mean()

                val_loss1 += loss_1.item()
                val_loss2 += loss_2.item()
                val_loss3 += loss_3.item()


                if not val_images_flag:
                    val_images_gt, val_images_fake = torch.zeros_like(B_spec[0]).unsqueeze(0), torch.zeros_like(predicted_noise[0]).unsqueeze(0)
                    for i, timestep in enumerate(timesteps):
                        reconstructed_spectrogram = model.scheduler.step(predicted_noise[i], timestep, noisy_spectrogram[i]).pred_original_sample
                        val_images_gt = torch.cat((val_images_gt, B_spec[i].unsqueeze(0)), dim=0)
                        val_images_fake = torch.cat((val_images_fake, reconstructed_spectrogram.unsqueeze(0)), dim=0)
                    val_images_flag = True


        val_l1_tensor = torch.tensor(val_loss1, device=device)
        val_l2_tensor = torch.tensor(val_loss2, device=device)
        val_l3_tensor = torch.tensor(val_loss3, device=device)
        global_l1 = val_l1_tensor / (len(val_loader))
        global_l2 = val_l2_tensor / (len(val_loader))
        global_l3 = val_l3_tensor / (len(val_loader))



        writer.add_scalar("Loss/Val_L1 Loss", global_l1, epoch)
        writer.add_scalar("Loss/Val_T60 Loss", global_l2, epoch)
        writer.add_scalar("Loss/Val_LPIPS Loss", global_l3, epoch)
        print(f"Epoch {epoch}, Total Validation Loss: {global_l1 + global_l2 + global_l3}")


        if val_images_flag:
            gt_spec, gen_spec = val_images_gt, val_images_fake
            writer.add_image("Validation/GT_Spectrogram", make_grid(gt_spec.cpu(), normalize=True), epoch)
            writer.add_image("Validation/Generated_Spectrogram", make_grid(gen_spec.cpu(), normalize=True), epoch)

        # Save checkpoints
        if (epoch + 1) % (args.epochs // 5) == 0 or global_l2 < best_val_loss:
            checkpoint_path = os.path.join(
                args.checkpoint_dir,
                args.version,
                f"{'best_val_checkpoint' if global_l2 < best_val_loss else f'epoch_{epoch + 1}_checkpoint'}.pth"
            )

            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_loss": min(global_l2, best_val_loss),
            }, checkpoint_path)
            if global_l2 < best_val_loss:
                best_val_loss = global_l2

    #writer.close()

# Argument parser
def main(args):
    checkpoint_folder = os.path.join(args.checkpoint_dir,args.version)
    if not os.path.isdir(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    # Device setup
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and DataLoader
    train_dataset = RIRDDMDataset(dataroot=args.data_dir, device=device, phase="train")
    val_dataset = RIRDDMDataset(dataroot=args.data_dir, device=device, phase="val")

    # Step 4 for DDP - Data Sampler as DistributedSampler and load it to DataLoader with sampler argument.


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=False)

    # Model, optimizer, and loss
    model = ConditionalDDPM(
        noise_channels=1, conditional_channels=1, embedding_dim=512, image_size=512, num_train_timesteps = NUM_TRAIN_TIMESTEPS
    ).to(device)

    num_gpus = torch.cuda.device_count()
    total_params = count_trainable_parameters(model)
    print(f"Total {num_gpus} GPUs available.")
    print(f"Total number of trainable parameters: {total_params}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=ADAM_BETA, eps=ADAM_EPS)
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, step / warmup_steps) * decay_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    criterion = nn.MSELoss()

    lpips_loss = lpips.LPIPS(net='vgg') # LPIPS loss for perceptual similarity
    lpips_loss.to(device)

    # Load from checkpoint if specified
    start_epoch = 0
    best_val_loss = float("inf")
    if args.from_pretrained:
        checkpoint = torch.load(args.from_pretrained, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]

    train_model(model=model, optimizer=optimizer, scheduler=scheduler, criterion=criterion, lpips_loss=lpips_loss, train_loader=train_loader, val_loader=val_loader, device=device, start_epoch=start_epoch, best_val_loss=best_val_loss, args=args)

if __name__ == "__main__":
    parser = ArgumentParser()
    #parser.add_argument("--data_dir", type=str, default="./datasets_subset_complete", help="Path to the dataset.")
    parser.add_argument("--data_dir", type=str, default="./datasets", help="Path to the dataset.")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--version", type=str, default="trial_06", help="The current training version of this model.")
    parser.add_argument("--from_pretrained", type=str, default=None, help="Path to a checkpoint to resume training.")
    parser.add_argument("--log_dir", type=str, default="./logs", help="TensorBoard log directory.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=50, help="Total number of epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--t60_ratio", type=float, default=0.5, help="The ratio between broadband t60 loss and octave-band split t60 loss.")
    args = parser.parse_args()

    main(args)
