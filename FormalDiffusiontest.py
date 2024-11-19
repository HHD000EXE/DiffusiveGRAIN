import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import math
import matplotlib.pyplot as plt

# Dataset Class
class ImageDataset(Dataset):
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.transform = Compose([Resize((256, 256)), ToTensor()])
        self.input_files = os.listdir(input_folder)
        self.output_files = os.listdir(output_folder)

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_image = Image.open(os.path.join(self.input_folder, self.input_files[idx])).convert("RGB")
        output_image = Image.open(os.path.join(self.output_folder, self.output_files[idx])).convert("RGB")
        input_image = self.transform(input_image)
        output_image = self.transform(output_image)
        return input_image, output_image

# UNet with Timestep Conditioning
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 512)
        )

        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(3 + 3 + 512, 64, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.MaxPool2d(2),
                                  nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.MaxPool2d(2),
                                  nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())
        self.middle = nn.Sequential(nn.MaxPool2d(2),
                                    nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(512, 512, 3, padding=1), nn.ReLU())

        # Decoder
        self.dec3 = nn.Sequential(nn.Conv2d(512 + 256, 256, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())
        self.dec2 = nn.Sequential(nn.Conv2d(256 + 128, 128, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.dec1 = nn.Sequential(nn.Conv2d(128 + 64, 64, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(64, 3, 3, padding=1), nn.Sigmoid())

    def forward(self, x, t, condition):
        # Embed timestep
        t_embed = self.time_embed(t.view(-1, 1).float())  # Shape: [batch_size, 512]
        t_embed = t_embed.view(x.size(0), 512, 1, 1).expand(x.size(0), 512, x.size(2), x.size(3))

        # Concatenate timestep embedding, noisy input, and condition image
        x = torch.cat((x, t_embed, condition), dim=1)  # Shape: [batch_size, 3+512+3, 256, 256]

        # UNet forward pass
        enc1_out = self.enc1(x)
        enc2_out = self.enc2(enc1_out)
        enc3_out = self.enc3(enc2_out)
        middle_out = self.middle(enc3_out)

        dec3_out = nn.functional.interpolate(middle_out, scale_factor=2, mode='nearest')
        dec3_out = torch.cat((dec3_out, enc3_out), dim=1)
        dec3_out = self.dec3(dec3_out)

        dec2_out = nn.functional.interpolate(dec3_out, scale_factor=2, mode='nearest')
        dec2_out = torch.cat((dec2_out, enc2_out), dim=1)
        dec2_out = self.dec2(dec2_out)

        dec1_out = nn.functional.interpolate(dec2_out, scale_factor=2, mode='nearest')
        dec1_out = torch.cat((dec1_out, enc1_out), dim=1)
        return self.dec1(dec1_out)

# Diffusion Loss
def diffusion_loss_with_reconstruction(model, x_0, t, alpha_cumprod):
    # Step 1: Sample random noise
    noise = torch.randn_like(x_0)

    # Step 2: Add noise to the clean image
    sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod[t.long()]).view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod[t.long()]).view(-1, 1, 1, 1)
    x_t = sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise

    # Step 3: Predict the noise using the model
    predicted_noise = model(x_t, t, x_0)

    # Step 4: Reconstruct the clean image
    reconstructed_x0 = (x_t - sqrt_one_minus_alpha_cumprod * predicted_noise) / sqrt_alpha_cumprod

    # Step 5: Compute the noise prediction loss
    loss = nn.functional.mse_loss(predicted_noise, noise)

    return loss, reconstructed_x0


# Noise Schedule
def prepare_noise_schedule(T):
    beta_start = 0.0001
    beta_end = 0.02
    beta = torch.linspace(beta_start, beta_end, T)
    alpha = 1.0 - beta
    alpha_cumprod = torch.cumprod(alpha, dim=0)
    return beta, alpha, alpha_cumprod

# Sampling
@torch.no_grad()
def sample(model, condition, T, alpha_cumprod):
    x_t = torch.randn((1, 3, 256, 256), device=device)  # Start with random noise
    for t in range(T - 1, -1, -1):
        t_tensor = torch.tensor([t], device=x_t.device).float()
        predicted_noise = model(x_t, t_tensor, condition)
        beta_t = beta[t]
        alpha_t = alpha[t]
        alpha_cumprod_t = alpha_cumprod[t]
        x_t = (x_t - beta_t * predicted_noise / (1 - alpha_cumprod_t).sqrt()) / alpha_t.sqrt()
        if t > 0:
            x_t += beta_t.sqrt() * torch.randn_like(x_t)  # Add noise
    return x_t

# Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = 1000
beta, alpha, alpha_cumprod = prepare_noise_schedule(T)
alpha_cumprod = alpha_cumprod.to(device)

# Load Dataset
train_dataset = ImageDataset("train", "label")
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# Initialize Model
model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
outputvisual = [500, 1000, 1999]

# Training
for epoch in range(2000):  # Adjust epochs as needed
    for x_0, condition in train_loader:
        x_0, condition = x_0.to(device), condition.to(device)
        t = torch.randint(0, T, (x_0.size(0),), device=device)

        # Compute loss and reconstruct clean image
        loss, reconstructed_x0 = diffusion_loss_with_reconstruction(model, x_0, t, alpha_cumprod)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Visualize a batch of results (optional)
        if epoch in outputvisual:
            clean_image = condition[0].permute(1, 2, 0).detach().cpu().numpy()
            noisy_image = (reconstructed_x0[0].permute(1, 2, 0).detach().cpu().numpy())
            input_image = x_0[0].permute(1, 2, 0).detach().cpu().numpy()

            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.title("Input (Condition)")
            plt.imshow(input_image)
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.title("Reconstructed Image")
            plt.imshow(noisy_image)
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.title("Ground Truth (Target)")
            plt.imshow(clean_image)
            plt.axis("off")

            plt.show()

    print(f"Epoch {epoch}: Loss {loss.item()}")


# Generate and Visualize Sample
@torch.no_grad()
def visualize_sample(model, condition_path, T, alpha_cumprod):
    transform = Compose([Resize((256, 256)), ToTensor()])
    condition = Image.open(condition_path).convert("RGB")
    condition = transform(condition).unsqueeze(0).to(device)

    output_image = sample(model, condition, T, alpha_cumprod)
    output_image = output_image.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu()

    plt.imshow(output_image)
    plt.axis("off")
    plt.show()

# visualize_sample(model, "train/20240418_x02y12_depth_2.png", T, alpha_cumprod)
