import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from model import Cnn


# Configure training device
device = torch.device('cpu:0' if torch.cuda.is_available() else 'cpu')

# Load MNIST dataset
data = datasets.MNIST(root='data', train=True, transform=ToTensor(), download=False)

# Initialize data loader
loader = DataLoader(data, batch_size=64, shuffle=True)

# Initialize neural model
cnn = Cnn()
cnn.train()

# Initialize loss function
loss_func = nn.CrossEntropyLoss()

# Initialize optimizer
optimizer = Adam(cnn.parameters(), lr=0.01)

n_epochs = 20
step_counter = 0
loss_log = []
step_log = []

for epoch in range(n_epochs):
    for images, labels in loader:
        # Forward
        ouputs = cnn(images.to(device))
        loss = loss_func(ouputs, labels.to(device))

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step_counter += 1
        if step_counter % 100 == 0:
            step_log.append(step_counter)
            loss_log.append(loss.item())
            print(f'Step: {step_counter}, Loss: {loss.item():.4f}')

torch.save(cnn.state_dict(), 'data_holder/cnn_model.pt')

# plt.plot(step_log, loss_log)
# plt.show()








