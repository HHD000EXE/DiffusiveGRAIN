import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.utils import save_image

import matplotlib.pyplot as plt
import os
import random
from PIL import Image


# -----------------------
#  Dataset
# -----------------------
class TrainDataset(Dataset):
    def __init__(self, image_folder, label_folder, transform):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform
        self.image_files = sorted(os.listdir(image_folder))
        self.label_files = sorted(os.listdir(label_folder))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.image_files[idx])
        label_name = os.path.join(self.label_folder, self.label_files[idx])

        image = Image.open(img_name).convert("RGB")
        label = Image.open(label_name).convert("RGB")

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label


# -----------------------
#  UNet Model
# -----------------------
def double_conv(in_channels, out_channels):
    """
    Two consecutive 3x3 conv + ReLU.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Encoder ---
        in_ch = in_channels
        for feature in features:
            self.downs.append(double_conv(in_ch, feature))
            in_ch = feature

        # --- Bottleneck ---
        self.bottleneck = double_conv(features[-1], features[-1] * 2)

        # --- Decoder ---
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    in_channels=feature * 2,
                    out_channels=feature,
                    kernel_size=2,
                    stride=2
                )
            )
            self.ups.append(double_conv(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # --- Encoder ---
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # --- Bottleneck ---
        x = self.bottleneck(x)

        # Reverse skip connections
        skip_connections = skip_connections[::-1]

        # --- Decoder ---
        for idx in range(0, len(self.ups), 2):
            up = self.ups[idx]
            conv = self.ups[idx + 1]
            x = up(x)  # Upsample

            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                diffY = skip_connection.size()[2] - x.size()[2]
                diffX = skip_connection.size()[3] - x.size()[3]
                skip_connection = skip_connection[:, :, diffY // 2 : skip_connection.size()[2] - diffY // 2,
                                                  diffX // 2 : skip_connection.size()[3] - diffX // 2]

            x = torch.cat((skip_connection, x), dim=1)
            x = conv(x)

        x = self.final_conv(x)
        return x


# -----------------------
#  Utility Functions
# -----------------------
def denormalize(tensor):
    """
    If your images are scaled to [-1, 1], this maps them back to [0, 1].
    """
    return (tensor * 0.5) + 0.5

def save_samples(epoch, model, dataset, device, prefix, num_samples=20, folder="output(Robot)"):
    """
    Generate 'num_samples' random samples from the given 'dataset',
    run them through the model, and save input, prediction, and label side by side.

    prefix: 'train' or 'test' to indicate which dataset we're sampling from
    """
    model.eval()
    os.makedirs(f"{folder}/{prefix}_epoch_{epoch+1}", exist_ok=True)

    # Random indices for sampling
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, label = dataset[idx]
            # Add batch dimension
            image_batch = image.unsqueeze(0).to(device)
            label_batch = label.unsqueeze(0).to(device)

            # Forward pass
            pred_batch = model(image_batch)

            # Detach & move to CPU
            image_np = image_batch.squeeze(0).cpu()
            pred_np  = pred_batch.squeeze(0).cpu()
            label_np = label_batch.squeeze(0).cpu()

            # Denormalize for saving
            image_d  = denormalize(image_np).clamp(0, 1)
            pred_d   = denormalize(pred_np).clamp(0, 1)
            label_d  = denormalize(label_np).clamp(0, 1)

            # Save them side by side
            # We can concatenate along width dimension (dim=2) or create a grid.
            # Here, we'll create a 3-image batch and let 'save_image' make a grid.
            images_to_save = torch.stack([image_d, pred_d, label_d], dim=0)
            save_path = f"{folder}/{prefix}_epoch_{epoch+1}/sample_{i}.png"

            # nrow=3 -> each row has 3 images, so effectively 1 row with (input, pred, label)
            save_image(images_to_save, save_path, nrow=3)


def plot_loss(train_losses, test_losses):
    """
    Plots the train and test losses vs. epoch.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()


# -----------------------
#   Main Training Script
# -----------------------
if __name__ == "__main__":
    # Hyperparameters
    IMG_SIZE = 64
    BATCH_SIZE = 1
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Transforms
    data_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        # Map [0,1] -> [-1,1]
        transforms.Lambda(lambda t: (t * 2) - 1),
    ])

    # Dataset & DataLoaders
    torch.manual_seed(0)
    train_dataset_folder = 'train_images(Robot-RawD)'  # Example paths
    label_dataset_folder = 'train_images(Robot-D)'     # Example paths

    dataset = TrainDataset(train_dataset_folder, label_dataset_folder, data_transforms)
    train_size = int(0.85 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    model = UNet(in_channels=3, out_channels=3).to(device)

    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    test_losses = []

    # -----------------------
    #  Training Loop
    # -----------------------
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_train_loss = 0.0

        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        epoch_train_loss = running_train_loss / len(train_dataloader)
        train_losses.append(epoch_train_loss)

        # -----------------------
        #  Evaluation / Testing
        # -----------------------
        model.eval()
        running_test_loss = 0.0

        with torch.no_grad():
            for images, labels in test_dataloader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_test_loss += loss.item()

        epoch_test_loss = running_test_loss / len(test_dataloader)
        test_losses.append(epoch_test_loss)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Train Loss: {epoch_train_loss:.4f} "
              f"Test  Loss: {epoch_test_loss:.4f}")

        # -----------------------
        # Generate sample images
        # -----------------------
        # For every 10 epochs, generate 20 samples for train & test
        if (epoch + 1) % 10 == 0:
            # Generate samples from training dataset
            save_samples(epoch, model, train_dataset, device, prefix="train", num_samples=20, folder="output(Robot)")

            # Generate samples from test dataset (if test set is non-empty)
            if len(test_dataset) > 0:
                save_samples(epoch, model, test_dataset, device, prefix="test", num_samples=20, folder="output(Robot)")

    # After training, plot the losses
    plot_loss(train_losses, test_losses)
