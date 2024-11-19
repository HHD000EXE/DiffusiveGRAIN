import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torch
from torchvision.transforms import Compose, Resize, ToTensor
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os


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
        img_name2 = os.path.join(self.label_folder, self.label_files[idx])
        image = Image.open(img_name)
        image2 = Image.open(img_name2)
        image = self.transform(image)
        image2 = self.transform(image2)

        return image.to(device), image2.to(device)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Encoder path
        self.enc1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Bottleneck
        self.middle = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Decoder path
        self.dec3 = nn.Sequential(
            nn.Conv2d(512 + 256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.dec2 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.dec1 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output in [0, 1] range
        )

    def forward(self, x, condition):
        # Concatenate input and condition along channel dimension
        x = torch.cat((x, condition), dim=1)

        # Encoder with skip connections
        enc1_out = self.enc1(x)
        enc2_out = self.enc2(enc1_out)
        enc3_out = self.enc3(enc2_out)

        # Bottleneck
        middle_out = self.middle(enc3_out)

        # Decoder with skip connections and upsampling
        dec3_out = F.interpolate(middle_out, scale_factor=2, mode='nearest')
        dec3_out = torch.cat((dec3_out, enc3_out), dim=1)
        dec3_out = self.dec3(dec3_out)

        dec2_out = F.interpolate(dec3_out, scale_factor=2, mode='nearest')
        dec2_out = torch.cat((dec2_out, enc2_out), dim=1)
        dec2_out = self.dec2(dec2_out)

        dec1_out = F.interpolate(dec2_out, scale_factor=2, mode='nearest')
        dec1_out = torch.cat((dec1_out, enc1_out), dim=1)
        dec1_out = self.dec1(dec1_out)

        return dec1_out


# Define function to add noise
def add_noise(x, amount):
    """Corrupt the input `x` by mixing it with noise according to `amount`."""
    noise = torch.rand_like(x)
    amount = amount.view(-1, 1, 1, 1)  # Sort shape so broadcasting works
    return x * (1 - amount) + noise * amount


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


def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    train_error, train_total, train_total_loss = 0, 0, 0
    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)
        # targets = torch.abs(targets - data).to(device)
        optimizer.zero_grad()
        noise_image = add_noise(data, noise_level).to(device)
        outputs = model(data, noise_image)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    print("loss:", loss)

    return data, outputs, targets


def evaluate(model, test_loader, criterion):
    model.eval()
    test_error, test_total_loss, test_total = 0, 0, 0
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)

            test_total += 1
            try:
                new_error = outputs - targets
                test_error = test_error + torch.abs(new_error)
            except:
                test_error = test_error
            test_total_loss += loss

        # Update total
        test_epoch_accuracy = test_error / test_total
        test_ave_loss = test_total_loss / test_total

    return test_epoch_accuracy, test_ave_loss


if __name__ == "__main__":
    # Check if CUDA is available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load images and convert to tensors
    image_folder = "train"  # Replace with your actual path
    label_folder = "label"  # Replace with your actual path

    transform = Compose([
        Resize((256, 256)),  # Resize images to a standard size
        ToTensor()])

    train_dataset = TrainDataset(image_folder, label_folder)
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)

    # Model, optimizer, and loss
    model = UNet().to(device)  # Move model to GPU
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    weight_zero = 0.1  # Weight for samples with label [0, y] in the first row
    weight_non_zero = 50  # Weight for samples with non-zero label in the first row
    criterion = nn.MSELoss()
    # criterion = WeightedMSELoss(weight_zero, weight_non_zero)

    # Training loop
    epochs = 2000
    show_epoch = [500, 1000, 1999]
    noise_level = torch.tensor([0.1], device=device)  # Ensure noise level tensor is on the correct device

    for epoch in range(epochs):
        print("epoch:", epoch)
        input_image, test_output, target_image = train_epoch(model, train_loader, criterion, optimizer)
        # test_epoch_accuracy, test_ave_loss = evaluate(model, train_loader, criterion)
        if epoch in show_epoch:
            # Convert the output tensor to a PIL image for visualization
            to_pil = transforms.ToPILImage()
            output_image = to_pil(test_output.squeeze(0).cpu())  # Move to CPU for display

            # Display the result
            plt.figure(figsize=(8, 8))
            plt.subplot(1, 3, 1)
            plt.title("Input Image")
            plt.imshow(input_image.squeeze(0).permute(1, 2, 0).cpu())

            plt.subplot(1, 3, 2)
            plt.title("Target Image")
            plt.imshow(target_image.squeeze(0).permute(1, 2, 0).cpu())

            plt.subplot(1, 3, 3)
            plt.title("Model Output")
            plt.imshow(output_image)
            plt.show()
