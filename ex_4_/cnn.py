import os
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch
import torchvision
from tqdm import tqdm
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt


class ResNet18(nn.Module):
    def __init__(self, pretrained=False, probing=False):
        super(ResNet18, self).__init__()
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
            self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
            self.resnet18 = resnet18(weights=weights)
        else:
            self.transform = transforms.Compose(
                [transforms.Resize((224, 224)), transforms.ToTensor()])
            self.resnet18 = resnet18()
        in_features_dim = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Identity()
        if probing:
            for name, param in self.resnet18.named_parameters():
                param.requires_grad = False
        self.logistic_regression = nn.Linear(in_features_dim, 1)

    def forward(self, x):
        features = self.resnet18(x)
        ### YOUR CODE HERE ###
        x = self.logistic_regression(features)
        return x


def get_loaders(path, transform, batch_size):
    """
    Get the data loaders for the train, validation and test sets.
    :param path: The path to the 'whichfaceisreal' directory.
    :param transform: The transform to apply to the images.
    :param batch_size: The batch size.
    :return: The train, validation and test data loaders.
    """
    train_set = torchvision.datasets.ImageFolder(
        root=os.path.join(path, 'train'), transform=transform)
    val_set = torchvision.datasets.ImageFolder(
        root=os.path.join(path, 'val'), transform=transform)
    test_set = torchvision.datasets.ImageFolder(
        root=os.path.join(path, 'test'), transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def compute_accuracy(model, data_loader, device):
    """
    Compute the accuracy of the model on the data in data_loader
    :param model: The model to evaluate.
    :param data_loader: The data loader.
    :param device: The device to run the evaluation on.
    :return: The accuracy of the model on the data in data_loader
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for (imgs, labels) in data_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs).squeeze()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += len(labels)
            correct += (predicted == labels).sum().item()
    return correct / total


def run_training_epoch(model, criterion, optimizer, train_loader, device):
    """
    Run a single training epoch
    :param model: The model to train
    :param criterion: The loss function
    :param optimizer: The optimizer
    :param train_loader: The data loader
    :param device: The device to run the training on
    :return: The average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    for (imgs, labels) in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs).squeeze()
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)


# Set the random seed for reproducibility
torch.manual_seed(0)

### UNCOMMENT THE FOLLOWING LINES TO TRAIN THE MODEL ###
# From Scratch
# model = ResNet18(pretrained=False, probing=False)
# Linear probing
# model = ResNet18(pretrained=True, probing=True)
# Fine-tuning
model = ResNet18(pretrained=True, probing=False)

transform = model.transform
batch_size = 32
num_of_epochs = 1
learning_rate = 0.0001
path = './whichfaceisreal'  # For example '/cs/usr/username/whichfaceisreal/'
train_loader, val_loader, test_loader = get_loaders(
    path, transform, batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
# Define the loss function and the optimizer
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Train the model

# Train the model
for epoch in range(num_of_epochs):
    # Run a training epoch
    loss = run_training_epoch(
        model, criterion, optimizer, train_loader, device)
    # Compute the accuracy
    train_acc = compute_accuracy(model, train_loader, device)
    # Compute the validation accuracy
    val_acc = compute_accuracy(model, val_loader, device)
    print(
        f'Epoch {epoch + 1}/{num_of_epochs}, Loss: {loss:.4f}, Val accuracy: {val_acc:.4f}')
    # Stopping condition
    if(val_acc > 0.9):
        break

# Compute the test accuracy
test_acc = compute_accuracy(model, test_loader, device)
for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)
    outputs = model(images).squeeze()
    predicted = (torch.sigmoid(outputs) > 0.5).float()
    misclassified_indices = (predicted != labels).nonzero()[:, 0]

    for idx in misclassified_indices:
        image = images[idx].cpu()
        true_label = labels[idx].cpu().item()
        predicted_label = predicted[idx].cpu().item()

        # Convert the image tensor to numpy array and transpose it to match the format expected by matplotlib
        image_np = image.numpy().transpose((1, 2, 0))

        # Plot the misclassified image
        plt.imshow(image_np)
        plt.title(
            f'True Label: {true_label}, Predicted Label: {predicted_label}')
        plt.axis('off')
        plt.show()
