import torch
import matplotlib.pyplot as plt

from torchvision import datasets
from torchvision.transforms import ToTensor

# Configure training device
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Load MNIST dataset
train_data = datasets.MNIST(root='data', train=True, transform=ToTensor(), download=False)
test_data = datasets.MNIST(root='data', train=False, transform=ToTensor(), download=False)

# Print dataset sumarry
print('\nData used for training')
print(train_data)
print(train_data.data.shape)
print(train_data.targets.shape)
print(train_data.targets[:10])

print('\nData used for testing')
print(test_data)
print(test_data.data.shape)
print(test_data.targets.shape)

# Plot one sample in training data
img_id = 0
plt.imshow(train_data.data[0], cmap='gray')
plt.title(f'{train_data.targets[img_id]}')
plt.show()

# Plot multiple images in training data
figure = plt.figure(figsize=(10, 8))
cols, rows = 5, 5

for i in range(1, cols * rows + 1):
    im_id = torch.randint(len(train_data), size=(1,)).item()
    image, label = train_data[im_id]
    # print(type(label))
    figure.add_subplot(rows, cols, i)
    plt.title(f'{label}')
    plt.axis('off')
    plt.imshow(image.squeeze(), cmap='gray')

plt.show()




