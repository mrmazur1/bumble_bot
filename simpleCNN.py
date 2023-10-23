from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import torch
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder


class myTransform():
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((192, 192)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomGrayscale(0.2),
            transforms.ColorJitter(brightness=0.2, saturation=0.2, hue=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def transform_input(self, image):
        ret = self.transform(image)
        return ret

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 96 * 96, 512)  # Adjust input size based on your image size
        self.fc2 = nn.Linear(512, 2)  # Adjust output size based on your task

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def train_model(self, output_dir, data_directory, batch_size=4):
        # Define your ImageFolder dataset
        transform = myTransform().transform
        dataset = ImageFolder(data_directory, transform=transform)

        # Specify the percentage for the validation set
        validation_split = 0.2  # 20% of the data for validation

        # Calculate the sizes for training and validation sets
        num_data = len(dataset)
        num_validation = int(validation_split * num_data)
        num_training = num_data - num_validation

        # Use random_split to split the dataset
        train_dataset, val_dataset = random_split(dataset, [num_training, num_validation])

        # Create DataLoader instances for training and validation
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
