import torch.nn as nn
import torch.nn.functional as F


class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.conv_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(32 * 7 * 7, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # Conv 1 -> Nonlinearity activation -> pooling
        x = self.conv_1(x)
        x = F.relu(x)
        x = self.pool(x)

        # Conv 2 -> Nonlinearity activation -> pooling
        x = self.conv_2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Flatten the output of the second convolution layer
        x = x.view(len(x), -1)

        # Put outputs through MLP
        x = self.fc(x)

        return x


class CnnOld(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Fully connected layer that outputs 10 classes
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)

        # Flatten the output of conv_2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)

        return self.fc(x)


