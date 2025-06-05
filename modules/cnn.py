import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=128):
        super(CNN, self).__init__()
        
        # First CNN block
        self.block1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )
        
        # Second CNN block
        self.block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        
        # Third CNN block
        self.block3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )
        
        # Max pooling layer
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        
        # Calculate the size after max pooling
        # Input size: 1 x 1335
        # After 3 blocks and max pooling: 256 x (1335/2)
        self.fc = nn.Linear(256 * (1335 // 2), num_classes)
        
    def forward(self, x):
        # Input shape: (batch_size, 1, 1335)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.maxpool(x)
        
        # Flatten the output for the linear layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

