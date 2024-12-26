import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNMLP(nn.Module):
    def __init__(self, num_out=3, input_channels=4):
        """
        Args:
            num_out (int): Dimension of the output. In your case, 3 (x, y, orientation).
            input_channels (int): Number of channels in the input image.
                                 By default 4 (3 RGB + 1 Depth).
        """
        super(CNNMLP, self).__init__()

        # --- CNN Feature Extractor ---
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # After these conv layers, the image is downsampled.
        # If the original size is (H, W), after two stride=2 layers, size ~ (H/4, W/4).
        # The output channels are 64. So the flattened dimension = 64 * (H/4) * (W/4).

        # --- MLP Layers ---
        # We'll reduce the feature dimension to something smaller before the final output.
        # For example, if input image is 64x64 after 2 downsamples -> 16x16, after 3 downsamples -> 8x8.
        # So flattened dimension might be 64 * 8 * 8 = 4096, for instance, if input is 64x64.
        # Adjust the dimension or add adaptive pooling if necessary.

        self.fc1 = nn.Linear(64 * 8 * 8, 256)  # adjust shape if your input size differs
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_out)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): A batch of combined images of shape (B, 4, H, W).
                              Where B is batch size, 4 is channels (3 RGB + 1 Depth).

        Returns:
            torch.Tensor: A tensor of shape (B, num_out), e.g. (B, 3).
        """
        # CNN feature extraction
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 16, H, W)
        x = F.relu(self.bn2(self.conv2(x)))  # (B, 32, H/2, W/2)
        x = F.relu(self.bn3(self.conv3(x)))  # (B, 64, H/4, W/4)

        # Flatten for MLP
        x = x.view(x.size(0), -1)  # shape: (B, 64*(H/4)*(W/4))

        # MLP part
        x = F.relu(self.fc1(x))  # (B, 256)
        x = F.relu(self.fc2(x))  # (B, 128)
        out = self.fc3(x)  # (B, 3) --> [x, y, orientation]

        return out


if __name__ == "__main__":
    # Example usage:

    # Suppose our input images are 64x64. Combined input = 4 channels (3 for RGB + 1 for Depth).
    batch_size = 2
    input_channels = 4
    height, width = 64, 64

    # Create a dummy input: shape = (B, C, H, W)
    dummy_input = torch.randn(batch_size, input_channels, height, width)

    # Instantiate the model
    model = CNNMLP(num_out=3, input_channels=input_channels)

    # Forward pass
    output = model(dummy_input)
    print("Output shape:", output.shape)
    print("Output:", output)
