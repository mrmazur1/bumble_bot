from torch import nn
import torch.nn.functional as F

import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

import torch
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib
from tqdm import tqdm

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

from myData import myData
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore", UserWarning)


class EarlyStopping:
    def __init__(self, patience=5, delta=0, checkpoint_path='best_checkpoint.pt'):
        self.patience = patience
        self.delta = delta
        self.checkpoint_path = checkpoint_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_checkpoint = None

    def __call__(self, val_loss, model, optimizer, epoch, bias):
        score = -val_loss
        bias = 0
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            print(f'Model checkpoint saved with validation loss: {val_loss}')
            #self.save_checkpoint(val_loss, model, optimizer, epoch)
            self.counter = 0
        else:
            self.counter += 1
            #self.load_best_checkpoint(model, optimizer, bias)
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model, optimizer, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }
        torch.save(checkpoint, self.checkpoint_path)
        print(f'Model checkpoint saved with validation loss: {val_loss}')
        self.best_checkpoint = self.checkpoint_path

    def load_best_checkpoint(self, model, optimizer, bias):
        if self.best_checkpoint is not None:
            checkpoint = torch.load(self.best_checkpoint)
            model.load_state_dict(checkpoint['model_state_dict'])

            # Update random_bias from the loaded checkpoint
            bias = nn.Parameter(torch.randn(1))

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f'Resumed training from epoch {checkpoint["epoch"]} with the loaded random_bias')


class Resnet_model(nn.Module):
    def __init__(self, data_directory, type=models.resnet18(pretrained=False)):
        super(Resnet_model, self).__init__()
        self.model = type
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model.to(self.device)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)
        self.model.to(self.device)

        # Create random_bias as a learnable parameter
        self.random_bias = nn.Parameter(torch.randn(1))
        self.data_directory = data_directory
        data = myData()
        self.labels = data.class_labels
        self.transform = data.transform


    def forward(self, x):
        resnet50_output = self.resnet50(x)
        # out_bias = resnet50_output+self.random_bias
        return resnet50_output

    def train(self, output_filename, optimizer, batch_size=16, epochs=8, lr = 0.001):

        dataset = datasets.ImageFolder(root=self.data_directory, transform=self.transform)
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

        criterion = nn.CrossEntropyLoss()
        #optimizerASGD = optim.ASGD(self.model.parameters(), weight_decay=1e-4)
        #optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
        early_stopping = EarlyStopping(patience=100, delta=0.0001, checkpoint_path=output_filename)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        # Training loop
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            self.model.train()  # Set the model to training mode
            running_loss = 0.0

            # Use tqdm for the loading bar
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch + 1}/{epochs}', unit='batch', leave=False)
            for batch_idx, (inputs, labels) in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            average_train_loss = running_loss / len(train_loader)
            train_losses.append(average_train_loss)
            scheduler.step()
            self.model.eval()  # Set the model to evaluation mode
            running_loss = 0.0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item()

            average_val_loss = running_loss / len(val_loader)
            val_losses.append(average_val_loss)
            early_stopping(average_val_loss, self.model, optimizer, epoch, self.random_bias)

            # print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {average_train_loss:.4f} - Val Loss: {average_val_loss:.4f}", end=" | ")

            if early_stopping.early_stop:
                print("Early stopping")
                epochs = len(val_losses)
                break

        plt.figure(1)
        valid = np.array(val_losses)
        train = np.array(train_losses)
        x = np.arange(1, epochs + 1)

        plt.title("Train and Validation Losses Over Epochs")
        plt.plot(x, valid, color='blue', label='Validation Loss')  # Validation loss
        plt.plot(x, train, color="red", label='Training Loss')  # Training loss
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"{output_filename}_losses.png")
        plt.close()

        #torch.save(early_stopping.best_checkpoint, "best_val_"+output_filename)
        torch.save(self.model.state_dict(), output_filename)
        return output_filename

class confusion_matrix_me():
    def __init__(self):
        pass

    def run(self,name, model, data_directory,batch_size=32):
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

        validation_split = 0.9
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

        cm = confusion_matrix(y_true=all_labels, y_pred=all_predictions, labels=[0, 1])

        print("total files checked: "+ str(len(train_dataset)))
        plt.figure(2, figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        vals = name.split('_')
        vals.pop()
        name = '_'.join(vals)
        plt.savefig("figure_"+name)
        plt.close()


