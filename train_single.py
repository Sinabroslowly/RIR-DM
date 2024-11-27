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
from scripts.model import ConditionalDDPM
from scripts.dataset import RIRDDMDataset
from scripts.util import compare_t60, estimate_t60, compare_t60_octave_bandwise, weighted_t60_err
from scripts.stft import STFT
import numpy as np

LAMBDAS = [1, 1, 1e-2, 1e-2] # LAMBDA multiplication for spectrogram reconstruction L1 loss, t60 error loss and lpips loss.
ADAM_BETA = (0.0, 0.99)
ADAM_EPS = 1e-8
NUM_TRAIN_TIMESTEPS = 80

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def octave_band_t60_error_loss(fake_spec, spec, device, t60_ratio=0.5):
    t60_err_fs = torch.Tensor([compare_t60(torch.pow(10,a).sum(-2).squeeze(), torch.pow(10,b).sum(-2).squeeze()) for a, b in zip(spec, fake_spec)]).to(device).mean()
    t60_errs = torch.Tensor([compare_t60_octave_bandwise(a, b) for a, b in zip(spec, fake_spec)]).to(device)
    t60_err_bs = weighted_t60_err(t60_errs)
    return ((1-t60_ratio)*t60_err_fs + t60_ratio*t60_err_bs)

def train_model(model, optimizer, criterion, lpips_loss, train_loader, val_loader, device, start_epoch, best_val_loss, args):

    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.version))
    stft = STFT()

    # Training loop
    for epoch in range(start_epoch+1, args.epochs):
        model.train()
        train_loss_total = 0
        train_loss_1 = 0
        train_loss_2 = 0
        train_loss_3 = 0
        train_loss_4 = 0
        for B_spec, text_embedding, image_embedding, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            # Split E_embedding into text and image embeddings

            B_spec = B_spec.to(device)  # Target spectrogram
            text_embedding = text_embedding.to(device)
            image_embedding = image_embedding.to(device)

            # Scheduler timestep
            noise = torch.randn_like(B_spec).to(device)
            timesteps = torch.randint(0, model.scheduler.config.num_train_timesteps, (B_spec.size(0),), device=device)
            #timesteps = torch.rand(B_spec.size(0), device=device)
            
            # Forward pass
            optimizer.zero_grad()
            fake_spec = torch.zeros_like(B_spec).to(device)
            #print(f"Shape of noisy_spectrogram: {noisy_spectrogram.shape}")

            noisy_spectrogram = model.scheduler.add_noise(B_spec, noise, timesteps)
            predicted_noise = model(noisy_spectrogram, timesteps, text_embedding, image_embedding)

            for i, timestep in enumerate(timesteps):
                
                # Denoising
                fake_spec[i] = model.scheduler.step(predicted_noise[i], timestep, noisy_spectrogram[i]).pred_original_sample

            loss_1 = criterion(noise, predicted_noise) # Reconstruction Loss between GT noise and added noise.
            loss_2 = criterion(B_spec, fake_spec) # Reconstruction Loss between GT spectrogram and denoised spectrogram from Gaussian Noise.
            loss_3 = octave_band_t60_error_loss(B_spec, fake_spec, device, args.t60_ratio)

            y_r = [stft.inverse(s.squeeze()) for s in B_spec]
            y_f = [stft.inverse(s.squeeze())for s in fake_spec]

            loss_4 = 1
            try:
                f = lambda x: pyroomacoustics.experimental.rt60.measure_rt60(x, 22050)
                t60_r = [f(y) for y in y_r if len(y)]
                t60_f = [f(y) for y in y_f if len(y)]
                loss_4 = np.mean([((t_b - t_a) / t_a) for t_a, t_b in zip(t60_r, t60_f)])
            except:
                pass

            loss = LAMBDAS[0] * loss_1 + LAMBDAS[1] * loss_2 + LAMBDAS[2] * loss_3 + LAMBDAS[3] * loss_4
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()
            train_loss_1 += loss_1.item()
            train_loss_2 += loss_2.item()
            train_loss_3 += loss_3.item()
            train_loss_4 += loss_4.item()
            
        train_loss_total_tensor = torch.tensor(train_loss_total, device=device)
        train_loss_1_tensor = torch.tensor(train_loss_1, device=device)
        train_loss_2_tensor = torch.tensor(train_loss_2, device=device)
        train_loss_3_tensor = torch.tensor(train_loss_3, device=device)
        train_loss_4_tensor = torch.tensor(train_loss_4, device=device)

        # Average the loss
        global_loss_total = train_loss_total_tensor / (len(train_loader))
        global_loss_1 = train_loss_1_tensor / (len(train_loader))
        global_loss_2 = train_loss_2_tensor / (len(train_loader))
        global_loss_3 = train_loss_3_tensor / (len(train_loader))
        global_loss_4 = train_loss_4_tensor / (len(train_loader))


        writer.add_scalar("Train/ Total Loss", global_loss_total, epoch)
        writer.add_scalar("Train/ L1_Loss - Noise", global_loss_1, epoch)
        writer.add_scalar("Train/ L1_Loss - Spec", global_loss_2, epoch)
        writer.add_scalar("Train/ OB_RT60 Loss", global_loss_3, epoch)
        writer.add_scalar("Train/ PRA_RT60 Loss", global_loss_4, epoch)
        print(f"Epoch {epoch}, Train Loss (Total): {global_loss_total}")

        # Validation loop
        model.eval()
        val_loss_1 = 0
        val_loss_2 = 0
        val_loss_3 = 0
        val_images_flag = False

        with torch.no_grad():
            for B_spec, text_embedding, image_embedding, _ in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
                # Split E_embedding into text and image embeddings

                B_spec = B_spec.to(device)  # Target spectrogram
                text_embedding = text_embedding.to(device)
                image_embedding = image_embedding.to(device)

                # Scheduler timestep
                noise = torch.randn_like(B_spec).to(device)
                timesteps = torch.randint(0, model.scheduler.config.num_train_timesteps, (B_spec.size(0),), device=device)

                # Forward pass
                noisy_spectrogram = model.scheduler.add_noise(B_spec, noise, timesteps)
                predicted_noise = model(noisy_spectrogram, timesteps, text_embedding, image_embedding)
                fake_spec = torch.zeros_like(B_spec).to(device)
                for i, timestep in enumerate(timesteps):    
                    # Denoising
                    fake_spec[i] = model.scheduler.step(predicted_noise[i], timestep, noisy_spectrogram[i]).pred_original_sample

                loss_1 = octave_band_t60_error_loss(B_spec, fake_spec, device, args.t60_ratio)

                y_r = [stft.inverse(s.squeeze()) for s in B_spec]
                y_f = [stft.inverse(s.squeeze())for s in fake_spec]

                loss_2 = 1
                try:
                    f = lambda x: pyroomacoustics.experimental.rt60.measure_rt60(x, 22050)
                    t60_r = [f(y) for y in y_r if len(y)]
                    t60_f = [f(y) for y in y_f if len(y)]
                    loss_2 = np.mean([((t_b - t_a) / t_a) for t_a, t_b in zip(t60_r, t60_f)])
                except:
                    pass

                loss_3 = lpips_loss(B_spec.repeat(1, 3, 1, 1), fake_spec.repeat(1, 3, 1, 1)).mean()

                val_loss_1 += loss_1.item()
                val_loss_2 += loss_2.item()
                val_loss_3 += loss_3.item()

                if not val_images_flag:
                    val_images_gt, val_images_fake = torch.zeros_like(B_spec[0]).unsqueeze(0), torch.zeros_like(predicted_noise[0]).unsqueeze(0)
                    for i, timestep in enumerate(timesteps):
                        reconstructed_spectrogram = model.scheduler.step(predicted_noise[i], timestep, noisy_spectrogram[i]).pred_original_sample
                        val_images_gt = torch.cat((val_images_gt, B_spec[i].unsqueeze(0)), dim=0)
                        val_images_fake = torch.cat((val_images_fake, reconstructed_spectrogram.unsqueeze(0)), dim=0)
                    val_images_flag = True


        val_l1_tensor = torch.tensor(val_loss_1, device=device)
        val_l2_tensor = torch.tensor(val_loss_2, device=device)
        val_l3_tensor = torch.tensor(val_loss_3, device=device)

        global_val_l1 = val_l1_tensor / (len(val_loader))
        global_val_l2 = val_l2_tensor / (len(val_loader))
        global_val_l3 = val_l3_tensor / (len(val_loader))



        writer.add_scalar("Validation/OB_T60 Loss", global_val_l1, epoch)
        writer.add_scalar("Validation/PRA_T60 Loss", global_val_l2, epoch)
        writer.add_scalar("Validation/LPIPS Loss", global_val_l3, epoch)

        print(f"Epoch {epoch}, Total Validation Loss: {global_val_l1 + global_val_l2 + global_val_l3}")


        if val_images_flag:
            gt_spec, gen_spec = val_images_gt, val_images_fake
            writer.add_image("Validation/GT_Spectrogram", make_grid(gt_spec.cpu(), normalize=True), epoch)
            writer.add_image("Validation/Generated_Spectrogram", make_grid(gen_spec.cpu(), normalize=True), epoch)

        # Save checkpoints
        if (epoch + 1) % (args.epochs // 5) == 0 or global_val_l2 < best_val_loss:
            checkpoint_path = os.path.join(
                args.checkpoint_dir,
                args.version,
                f"{'best_val_checkpoint' if global_val_l2 < best_val_loss else f'epoch_{epoch + 1}_checkpoint'}.pth"
            )
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_loss": min(global_val_l2, best_val_loss),
            }, checkpoint_path)
            if global_val_l2 < best_val_loss:
                best_val_loss = global_val_l2

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

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=False)

    # Model, optimizer, and loss
    model = ConditionalDDPM(
        noise_channels=1, embedding_dim=512, image_size=512, num_train_timesteps = NUM_TRAIN_TIMESTEPS
    ).to(device)

    num_gpus = torch.cuda.device_count()
    total_params = count_trainable_parameters(model)
    print(f"Total {num_gpus} GPUs available.")
    print(f"Total number of trainable parameters: {total_params}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=ADAM_BETA, eps=ADAM_EPS)
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, step / warmup_steps) * decay_rate)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    model.scheduler.set_timesteps(num_inference_steps=NUM_TRAIN_TIMESTEPS)

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

    train_model(model=model, optimizer=optimizer, criterion=criterion, lpips_loss=lpips_loss, train_loader=train_loader, val_loader=val_loader, device=device, start_epoch=start_epoch, best_val_loss=best_val_loss, args=args)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./datasets_subset_complete", help="Path to the dataset.")
    #parser.add_argument("--data_dir", type=str, default="./datasets", help="Path to the dataset.")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--version", type=str, default="trial_05", help="The current training version of this model.")
    parser.add_argument("--from_pretrained", type=str, default=None, help="Path to a checkpoint to resume training.")
    parser.add_argument("--log_dir", type=str, default="./logs", help="TensorBoard log directory.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=100, help="Total number of epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--t60_ratio", type=float, default=0.5, help="The ratio between broadband t60 loss and octave-band split t60 loss.")
    args = parser.parse_args()

    main(args)
