from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import torch
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

import torch
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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


class Trainer():
    def __init__(self):
        self.model = SimpleCNN()

    def train_model(self, output_filename, data_directory, batch_size=16, epochs=8):
        # Define your ImageFolder dataset
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        self.model.to(device=device)
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

        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Training loop
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            self.model.train()  # Set the model to training mode
            running_loss = 0.0

            for batch_idx, data in enumerate(train_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)  # Move data to the GPU
                optimizer.zero_grad()  # Zero the parameter gradients
                outputs = self.model(inputs)  # Forward pass
                loss = criterion(outputs, labels)  # Calculate the loss
                loss.backward()  # Backpropagation
                optimizer.step()  # Update weights

                running_loss += loss.item()
                print(f"Epoch [{epoch + 1}/{epochs}] "
                      f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                      f"Loss: {running_loss / 10:.4f}")

            # Calculate and store the average training loss for the epoch
            average_train_loss = running_loss / len(train_loader)
            train_losses.append(average_train_loss)

            # Validation loop (evaluate the model on the validation set)
            self.model.eval()  # Set the model to evaluation mode
            running_loss = 0.0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)  # Move data to the GPU

                    outputs = self.model(inputs)  # Forward pass
                    loss = criterion(outputs, labels)  # Calculate the loss

                    running_loss += loss.item()

            # Calculate and store the average validation loss for the epoch
            average_val_loss = running_loss / len(val_loader)
            val_losses.append(average_val_loss)

            # Print or log the training and validation losses for this epoch
            print(
                f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {average_train_loss:.4f} - Val Loss: {average_val_loss:.4f}")

        torch.save(self.model.state_dict(), output_filename)


class Resnet_model:
    def __init__(self):
        pass

    def train(self, output_filename, data_directory, batch_size=16, epochs=8):
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        dataset = datasets.ImageFolder(root=data_directory, transform=transform)

        # Specify the percentage for the validation set
        validation_split = 0.2  # 20% of the data for validation

        # Calculate the sizes for training and validation sets
        num_data = len(dataset)
        num_validation = int(validation_split * num_data)
        num_training = num_data - num_validation

        # Use random_split to split the dataset
        train_dataset, val_dataset = random_split(dataset, [num_training, num_validation])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        model = models.resnet18(pretrained=False)
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        model.to(device)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)  # Assuming num_classes is the number of classes in your dataset
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.adam(model.parameters(), lr=0.0005, momentum=0.9)

        # Training loop
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            model.train()  # Set the model to training mode
            running_loss = 0.0

            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                print(f"Epoch [{epoch + 1}/{epochs}] "
                      f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                      f"epoch Loss: {running_loss / (batch_idx + 1):.4f} "
                      f"Total Loss: {running_loss / epochs:.4f}")

            average_train_loss = running_loss / len(train_loader)
            train_losses.append(average_train_loss)

            model.eval()  # Set the model to evaluation mode
            running_loss = 0.0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item()

            average_val_loss = running_loss / len(val_loader)
            val_losses.append(average_val_loss)

            print(
                f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {average_train_loss:.4f} - Val Loss: {average_val_loss:.4f}")

        torch.save(model.state_dict(), output_filename)


class confusion_matrix_me():
    def __init__(self):
        pass

    def run(self, model, data_directory,batch_size=32):
        # Assuming your model is named 'model' and your test loader is 'test_loader'
        device = torch.device("cuda")
        model.to(device)
        model.eval()
        all_labels = []
        all_predictions = []

        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        validation_split = 0
        dataset = ImageFolder(data_directory, transform=transform)
        num_data = len(dataset)
        num_validation = int(validation_split * num_data)
        num_training = num_data - num_validation

        # Use random_split to split the dataset
        train_dataset, val_dataset = random_split(dataset, [num_training, num_validation])
        test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        class_labels= ['hot', 'not']

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        cm = confusion_matrix(y_true=all_labels, y_pred=all_predictions, labels=[1, 0])

        print("total files checked: "+ str(len(train_dataset)))
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig("figure")
        plt.show()


