import os
import json
import argparse
import numpy as np
import torch
import seaborn as sns
import pyroomacoustics
from tqdm import tqdm
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from accelerate import Accelerator
from sklearn.linear_model import LinearRegression
from scripts.dataset import RIRDDMDataset
from scripts.model import ConditionalDDPM
from scripts.stft import STFT

NUM_SAMPLE_STEPS = 30

def save_t60_analysis(examples, t60_err, t60_vals, output_dir, version):
    """
    Save T60 error analysis: Boxplot, JSON, and Scatter plot.
    """
    print(f"Length of t60_err: {len(t60_err)}")
    print(f"Length of examples: {len(examples)}")
    t60_err = np.array(t60_err) * 100  # Convert errors to percentages

    # Save T60 errors as a NumPy file
    np.save(os.path.join(output_dir, "t60_err.npy"), t60_err)

    # Create and save the boxplot for T60 error
    plt.figure(figsize=(4, 5))
    sns.boxplot(y=t60_err)
    plt.title(f"{version} Boxplot T60 Error")
    plt.savefig(os.path.join(output_dir, f"{version}_t60_boxplot.png"))
    plt.close()

    # Save T60 errors and absolute T60 values as a JSON file
    t60_dict = {example.split('/')[-1].split('.')[0]: t60_err[i] for i, example in enumerate(examples)}
    with open(os.path.join(output_dir, "t60.json"), "w") as json_file:
        json.dump(t60_dict, json_file, indent=4)

    # Scatter plot for absolute T60 difference
    t_a = [pair[0] for pair in t60_vals]  # Ground truth T60 values
    t_ba = [pair[1] - pair[0] for pair in t60_vals]  # Absolute T60 difference

    plt.figure(figsize=(6, 5))
    plt.scatter(t_a, t_ba, alpha=0.5, marker=".")
    
    # Linear regression
    t_a_np = np.array(t_a).reshape(-1, 1)
    t_ba_np = np.array(t_ba)
    linear_regressor = LinearRegression()
    linear_regressor.fit(t_a_np, t_ba_np)
    y_pred = linear_regressor.predict(t_a_np)
    slope, intercept = linear_regressor.coef_[0], linear_regressor.intercept_

    # Plot regression line
    plt.plot(t_a, y_pred, linestyle="dotted", color="red", label="Linear fit")
    equation_text = f"y = {slope:.2f}x + {intercept:.2f}"
    plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment="top", color="red")

    # Configure plot
    plt.xlabel("GT_RT60 (s)")
    plt.ylabel("Absolute RT60 Difference (s)")
    plt.ylim([-3, 3])
    plt.xlim([0, 4])
    plt.title(f"{version} Absolute RT60 Difference")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{version}_t60_tendency.png"))
    plt.close()

    print(f"T60 errors, boxplot, and scatter plot saved in {output_dir}")


def main():
    # Ensure GPU environment
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    accelerator = Accelerator()
    device = accelerator.device

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints", help="Path to pretrained model checkpoint.")
    parser.add_argument("--checkpoint_ver", type=str, default=None, help="Checkpoint version to load.")
    parser.add_argument("--data_dir", type=str, default="datasets", help="Dataset path.")
    parser.add_argument("--output_dir", type=str, default="./test_output_t60only", help="Directory for test outputs.")
    parser.add_argument("--version", type=str, default="trial_07", help="Experiment version.")
    #parser.add_argument("--final_step", type=bool, default="False", help="Sample from pure noise or allow intermediate sampling")
    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = os.path.join(args.output_dir, args.version)
    os.makedirs(output_dir, exist_ok=True)

    # Load the test dataset
    test_dataset = RIRDDMDataset(dataroot=args.data_dir, device=device, phase="test")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    # Load the Conditional DDPM model
    model = ConditionalDDPM(noise_channels=1, embedding_dim=512, image_size=512, num_train_timesteps=NUM_SAMPLE_STEPS).to(device)
    model.scheduler.set_timesteps(num_inference_steps=NUM_SAMPLE_STEPS, device=device)

    model, test_loader = accelerator.prepare(model, test_loader)

    # Load model checkpoint
    checkpoint = torch.load(os.path.join(args.checkpoints, args.version, args.checkpoint_ver), map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    stft = STFT()

    # Testing process
    test_examples = []
    t60_err = []
    t60_vals = []

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = model.scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = model.scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)

        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    with torch.no_grad():
        for B_spec, text_embedding, image_embedding, _, paths in tqdm(test_loader, desc=f"Test Validaiton:", leave=False):
            text_embedding = text_embedding.to(device)
            image_embedding = image_embedding.to(device)
            B_spec = B_spec.to(device)
            examples = [os.path.basename(s[:s.rfind(".")]) for s, _ in zip(*paths)]

            ### Generate noise and add to spectrogram (training phase) ###
            noise = torch.randn_like(B_spec).to(device)
            bsz = B_spec.shape[0]
            indices = torch.randint(0, model.scheduler.config.num_train_timesteps, (bsz,))
            timesteps = model.scheduler.timesteps[indices].to(device)
            noisy_spectrogram = model.scheduler.add_noise(B_spec, noise, timesteps)
            sigmas = get_sigmas(timesteps, len(noisy_spectrogram.shape), noisy_spectrogram.dtype)
            sigma_noisy_spectrogram = model.scheduler.precondition_inputs(noisy_spectrogram, sigmas)
            predicted_noise = model(sigma_noisy_spectrogram, timesteps, text_embedding, image_embedding)
            denoised_sample = model.scheduler.precondition_outputs(noisy_spectrogram, predicted_noise, sigmas)
            ##############################################################
            bsz = B_spec.shape[0]
            latent_noise = torch.randn(B_spec.shape, device=device) * model.scheduler.init_noise_sigma
            for t in model.scheduler.timesteps:
                model_input = model.scheduler.scale_model_input(latent_noise, t)
                predicted_noise = model(model_input, t, text_embedding, image_embedding)
                latent_noise = model.scheduler.step(predicted_noise, t, latent_noise).prev_sample

            denoised_sample = latent_noise
            
            # Reconstruct audio with istft.
            gt_audio = [stft.inverse(s) for s in B_spec]
            gen_audio = [stft.inverse(s) for s in denoised_sample]
            # Calculate T60 error
            t60_gt_audio = pyroomacoustics.experimental.rt60.measure_rt60(gt_audio, 22050)
            t60_gen_audio = pyroomacoustics.experimental.rt60.measure_rt60(gen_audio, 22050)
            
            try:
                t60_vals.append([t60_gt_audio, t60_gen_audio])
                t60_err.append((t60_gen_audio - t60_gt_audio) / t60_gt_audio)
            except:
                t60_err.append(np.nan)

            # Store example names
            #example_name = os.path.basename(paths[0][0])
            test_examples.extend(examples)

    # Save analysis
    save_t60_analysis(test_examples, t60_err, t60_vals, output_dir, args.version)


if __name__ == "__main__":
    main()
