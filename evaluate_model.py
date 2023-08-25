import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from model import Cnn


# Configure training device
device = torch.device('cpu:0' if torch.cuda.is_available() else 'cpu')

# Load MNIST dataset
data = datasets.MNIST(root='data', train=False, transform=ToTensor(), download=False)

# Initialize data loader
loader = DataLoader(data, batch_size=64, shuffle=False)

# Initialize neural model
cnn = Cnn()
cnn.eval()
cnn.load_state_dict(torch.load('data_holder/cnn_model.pt'))

n_samples = 0
n_correct = 0

for images, labels in loader:
    # Forward
    ouputs = cnn(images.to(device))
    predicted_labels = torch.argmax(ouputs, dim=1)
    n_samples += len(labels)
    n_correct += (predicted_labels == labels.to(device)).sum().item()

accuracy = 100.0 * n_correct / n_samples
print(f'Accuracy of the network: {accuracy} %')










