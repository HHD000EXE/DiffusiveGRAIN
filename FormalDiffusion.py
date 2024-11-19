import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader, Dataset
import os
import math
from PIL import Image
import matplotlib.pyplot as plt

# Dataset Class
class TrainDataset(Dataset):
    def __init__(self, image_folder, label_folder):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = Compose([Resize((256, 256)), ToTensor()])
        self.image_files = os.listdir(image_folder)
        self.label_files = os.listdir(label_folder)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.image_files[idx])
        label_name = os.path.join(self.label_folder, self.label_files[idx])
        image = Image.open(img_name).convert("RGB")
        label = Image.open(label_name).convert("RGB")
        image = self.transform(image)
        label = self.transform(label)
        return image, label

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
        self.enc1 = nn.Sequential(nn.Conv2d(3 + 512, 64, 3, padding=1), nn.ReLU(),
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

    def forward(self, x, t):
        # Embed time step
        t_embed = self.time_embed(t.view(-1, 1).float())  # Convert t to float
        t_embed = t_embed.view(1, 512, 1, 1).expand(1, 512, 256, 256)

        # Concatenate timestep embedding with input
        x = torch.cat((x, t_embed), dim=1)

        # UNet
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

# Noise Schedules
def prepare_noise_schedule(T):
    beta_start = 0.0
    beta_end = 0.00
    beta = torch.linspace(beta_start, beta_end, T)
    alpha = 1.0 - beta
    alpha_cumprod = torch.cumprod(alpha, dim=0)
    return beta, alpha, alpha_cumprod


class WeightedMSELoss(nn.Module):
    def __init__(self, weight_zero, weight_non_zero):
        super(WeightedMSELoss, self).__init__()
        self.weight_zero = weight_zero
        self.weight_non_zero = weight_non_zero

    def forward(self, input, target):
        mask_tensor = torch.where(abs(target) < 0.04, torch.tensor(weight_zero), torch.tensor(weight_non_zero))
        criterion = nn.MSELoss(reduction='none')  # Use 'none' to avoid reducing the loss yet
        mse_loss = criterion(input, target)
        weighted_loss = torch.mean(mask_tensor*mse_loss)

        return weighted_loss


# Function to generate and visualize sample input and output
def visualize_sample(model, image_path, T, alpha_cumprod):
    # Create a single sample input image
    transform = Compose([Resize((256, 256)), ToTensor()])
    sample_image = Image.open(image_path).convert("RGB")
    sample_image = transform(sample_image).unsqueeze(0).to(device)  # Shape: [1, 3, 256, 256]

    # Add random noise to simulate a noisy input
    t = torch.randint(0, T, (1,), device=device).float()
    noise = torch.randn_like(sample_image)
    sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod[t.long()]).view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod[t.long()]).view(-1, 1, 1, 1)
    noisy_image = sqrt_alpha_cumprod * sample_image + sqrt_one_minus_alpha_cumprod * noise

    # Pass the noisy input through the model
    output_image = model(noisy_image, t)

    # Detach tensors and clamp values to [0, 1]
    sample_image = sample_image.squeeze(0).permute(1, 2, 0).detach().cpu().clamp(0, 1)
    noisy_image = noisy_image.squeeze(0).permute(1, 2, 0).detach().cpu().clamp(0, 1)
    output_image = output_image.squeeze(0).permute(1, 2, 0).detach().cpu().clamp(0, 1)

    # Display the images
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(sample_image)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Noisy Image")
    plt.imshow(noisy_image)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Model Output")
    plt.imshow(output_image)
    plt.axis('off')

    plt.show()


# Training Setup
weight_zero = 0.1  # Weight for samples with label [0, y] in the first row
weight_non_zero = 50  # Weight for samples with non-zero label in the first row
criterion = WeightedMSELoss(weight_zero, weight_non_zero)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = 1000
beta, alpha, alpha_cumprod = prepare_noise_schedule(T)
alpha_cumprod = alpha_cumprod.to(device)

# Initialize and train the model
model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
train_dataset = TrainDataset("train", "label")
train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)

for epoch in range(100):
    for input, label in train_loader:
        input = input.to(device)
        t = torch.randint(0, T, (input.size(0),), device=device)
        output = model(input, t)
        loss = criterion(input, output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}: Loss {loss.item()}")

# Generate a sample visualization
visualize_sample(model, "train/20240418_x02y12_depth_2.png", T, alpha_cumprod)
