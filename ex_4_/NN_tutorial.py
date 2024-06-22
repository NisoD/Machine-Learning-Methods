import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from helpers import *


# def train_model(train_data, val_data, test_data, model, lr=0.001, epochs=50, batch_size=256):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
#     print('Using device:', device)

#     trainset = torch.utils.data.TensorDataset(torch.tensor(
#         train_data[['long', 'lat']].values).float(), torch.tensor(train_data['country'].values).long())
#     valset = torch.utils.data.TensorDataset(torch.tensor(
#         val_data[['long', 'lat']].values).float(), torch.tensor(val_data['country'].values).long())
#     testset = torch.utils.data.TensorDataset(torch.tensor(
#         test_data[['long', 'lat']].values).float(), torch.tensor(test_data['country'].values).long())

#     trainloader = torch.utils.data.DataLoader(
#         trainset, batch_size=batch_size, shuffle=True, num_workers=0)
#     valloader = torch.utils.data.DataLoader(
#         valset, batch_size=1024, shuffle=False, num_workers=0)
#     testloader = torch.utils.data.DataLoader(
#         testset, batch_size=1024, shuffle=False, num_workers=0)

#     criterion = nn.CrossEntropyLoss().to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#     train_accs = []
#     val_accs = []
#     test_accs = []
#     train_losses = []
#     val_losses = []
#     test_losses = []

#     for ep in range(epochs):
#         model.train()
#         pred_correct = 0
#         ep_loss = 0.
#         for i, (inputs, labels) in enumerate(tqdm(trainloader)):
#             #### YOUR CODE HERE ####
#             # perform a training iteration
#             inputs, labels = inputs.to(device), labels.to(device)
#             # move the inputs and labels to the device
#             optimizer.zero_grad()
#             # zero the gradients
#             outputs = model(inputs)
#             # forward pass
#             loss = criterion(outputs, labels)

#             # calculate the loss
#             loss.backward()
#             # backward pass
#             optimizer.step()
#             # update the weights

#             # name the model outputs "outputs"
#             # and the loss "loss"

#             #### END OF YOUR CODE ####

#             pred_correct += (torch.argmax(outputs, dim=1)
#                              == labels).sum().item()
#             ep_loss += loss.item()

#         train_accs.append(pred_correct / len(trainset))
#         train_losses.append(ep_loss / len(trainloader))

#         model.eval()
#         with torch.no_grad():
#             for loader, accs, losses in zip([valloader, testloader], [val_accs, test_accs], [val_losses, test_losses]):
#                 correct = 0
#                 total = 0
#                 ep_loss = 0.
#                 for inputs, labels in loader:
#                     #### YOUR CODE HERE ####

#                     # perform an evaluation iteration
#                     inputs, labels = inputs.to(device), labels.to(device)

#                     # move the inputs and labels to the device
#                     outputs = model(inputs)
#                     # forward pass
#                     loss = criterion(outputs, labels)
#                     # calculate the loss
#                     # name the model outputs "outputs"
#                     # and the loss "loss"
#                     #### END OF YOUR CODE ####

#                     ep_loss += loss.item()
#                     _, predicted = torch.max(outputs.data, 1)
#                     total += labels.size(0)
#                     correct += (predicted == labels).sum().item()

#                 accs.append(correct / total)
#                 losses.append(ep_loss / len(loader))

#         print('Epoch {:}, Train Acc: {:.3f}, Val Acc: {:.3f}, Test Acc: {:.3f}'.format(
#             ep, train_accs[-1], val_accs[-1], test_accs[-1]))

#     return model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses


def question1(train_data, val_data, test_data):
    learning_rates = [1, 0.01, 0.001, 0.00001]
    epochs = 2
    model_list = []
    model_list_losses = []
    model_list_accs = []
    output_dim = len(train_data['country'].unique())
    for lr in learning_rates:
        model = [nn.Linear(2, 16), nn.ReLU(),  # hidden layer 1
                 nn.Linear(16, 16), nn.ReLU(),  # hidden layer 2
                 nn.Linear(16, 16), nn.ReLU(),  # hidden layer 3
                 nn.Linear(16, 16), nn.ReLU(),  # hidden layer 4
                 nn.Linear(16, 16), nn.ReLU(),  # hidden layer 5
                 nn.Linear(16, 16), nn.ReLU(),  # hidden layer 6
                 nn.Linear(16, output_dim)  # output layer
                 ]
        model = nn.Sequential(*model)
        model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses = train_model(
            train_data, val_data, test_data, model, lr=lr, epochs=epochs, batch_size=256)
        model_list_losses.append((lr, train_losses, val_losses, test_losses))
        model_list_accs.append((lr, train_accs, val_accs, test_accs))
        model_list.append(model)
    # plot accuracies on val set per model
    plt.figure()
    plt.title('Validation Accuracies')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    for model in model_list_accs:
        plt.plot(model[1], label=f'lr={model[0]}')
    plt.legend()
    plt.show()

    # plot loss on val set per model
    plt.figure()
    plt.title('Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    for model in model_list_losses:
        plt.plot(model[2], label=f'lr={model[0]}')
    plt.legend()
    plt.show()

    return model_list_losses, model_list_accs, model_list


def build_nn(input_dim, output_dim=2, depth=6, width=16, batch_norm=False):
    """
    Builds a neural network architecture.

    Args:
      input_dim: The input dimension.
      output_dim: The output dimension.
      depth: The number of hidden layers.
      width: The number of neurons per hidden layer.
      batch_norm: Whether to use batch normalization.

    Returns:
      A neural network model.
    """

    model = []
    for _ in range(depth):
        model += [nn.Linear(input_dim, width), nn.ReLU()]
        if batch_norm:
            model += [nn.BatchNorm1d(width)]
        input_dim = width

    model += [nn.Linear(input_dim, output_dim)]

    return nn.Sequential(*model)


def get_epochs_losses(train_data, val_data, test_data):
    epoch_losses_dict = {}
    epochs = [1, 5, 10, 20, 50, 100]
    output_dim = len(train_data['country'].unique())
    model = build_nn(2, output_dim, depth=6, width=16, batch_norm=False)
    _, _, _, _, _, val_losses, _ = train_model(train_data, val_data, test_data, model, lr=0.001, epochs=100,
                                               batch_size=256)

    epochs_1 = val_losses[0]
    epochs_2 = [None] + val_losses[1:5]
    epochs_3 = [None] * 5 + val_losses[5:10]
    epochs_4 = [None] * 10 + val_losses[10:20]
    epochs_5 = [None] * 20 + val_losses[20:50]
    epochs_6 = [None] * 50 + val_losses[50:100]
    epoch_losses_dict = {1: epochs_1, 5: epochs_2,
                         10: epochs_3, 20: epochs_4, 50: epochs_5, 100: epochs_6}

    colors = ['b', 'g', 'r', 'c', 'm', 'y']  # List of colors for each epoch

    for epoch, losses in epoch_losses_dict.items():
        plt.plot(losses, label=f'Epoch={epoch}',
                 color=colors[epochs.index(epoch)])
    plt.title('Validation Losses vs. epochs for different epochs')
    plt.legend()
    plt.show()


def get_epochs_losses_no_batch_normal(train_data, val_data, test_data):
    epoch_losses_dict = {}
    epochs = [1, 5, 10, 20, 50, 100]
    output_dim = len(train_data['country'].unique())
    model = build_nn(2, output_dim, depth=6, width=16, batch_norm=True)
    _, _, _, _, _, val_losses, _ = train_model(train_data, val_data, test_data, model, lr=0.001, epochs=100,
                                               batch_size=256)
    epochs_1 = val_losses[0]
    epochs_2 = [None] + val_losses[1:5]
    epochs_3 = [None] * 5 + val_losses[5:10]
    epochs_4 = [None] * 10 + val_losses[10:20]
    epochs_5 = [None] * 20 + val_losses[20:50]
    epochs_6 = [None] * 50 + val_losses[50:100]
    epoch_losses_dict = {1: epochs_1, 5: epochs_2,
                         10: epochs_3, 20: epochs_4, 50: epochs_5, 100: epochs_6}

    colors = ['b', 'g', 'r', 'c', 'm', 'y']  # List of colors for each epoch

    for epoch, losses in epoch_losses_dict.items():
        plt.plot(losses, label=f'Epoch={epoch}',
                 color=colors[epochs.index(epoch)])
    plt.title('Validation Losses vs. epochs for different epochs')
    plt.legend()
    plt.show()


def plot_epochs_normal_vs_no_batch_normal(train_data, val_data, test_data):
    epoch_losses_dict = {}
    epochs = [1, 5, 10, 20, 50, 100]
    output_dim = len(train_data['country'].unique())
    model = build_nn(2, output_dim, depth=6, width=16, batch_norm=False)
    _, _, _, _, _, val_losses, _ = train_model(train_data, val_data, test_data, model, lr=0.001, epochs=100,
                                               batch_size=256)
    model2 = build_nn(2, output_dim, depth=6, width=16, batch_norm=True)
    _, _, _, _, _, val_losses2, _ = train_model(train_data, val_data, test_data, model2, lr=0.001, epochs=100,
                                                batch_size=256)
    epochs_1 = val_losses[0]
    epochs_2 = [None] + val_losses[1:5]
    epochs_3 = [None] * 5 + val_losses[5:10]
    epochs_4 = [None] * 10 + val_losses[10:20]
    epochs_5 = [None] * 20 + val_losses[20:50]
    epochs_6 = [None] * 50 + val_losses[50:100]
    epoch_losses_dict = {1: epochs_1, 5: epochs_2,
                         10: epochs_3, 20: epochs_4, 50: epochs_5, 100: epochs_6}
    colors = iter(['red', 'blue', 'green', 'orange', 'purple', 'black'])
    epochs_1_2 = val_losses2[0]
    epochs_2_2 = [None] + val_losses2[1:5]
    epochs_3_2 = [None] * 5 + val_losses2[5:10]
    epochs_4_2 = [None] * 10 + val_losses2[10:20]
    epochs_5_2 = [None] * 20 + val_losses2[20:50]
    epochs_6_2 = [None] * 50 + val_losses2[50:100]
    epoch_losses_dict2 = {1: epochs_1_2, 5: epochs_2_2,
                          10: epochs_3_2, 20: epochs_4_2, 50: epochs_5_2, 100: epochs_6_2}
    for epoch, losses in epoch_losses_dict.items():
        plt.plot(losses, label=f'Epoch={epoch}', color=colors.__next__())

    colors = iter(['red', 'blue', 'green', 'orange', 'purple', 'black'])
    for epoch, losses in epoch_losses_dict2.items():
        plt.plot(losses, label=f'Epoch={epoch}', color=colors.__next__())
    plt.title(
        'Validation Losses vs. epochs for different epochs with and without batch normal')
    plt.legend()
    plt.show()


def question4(train_data, val_data, test_data):
    batch_size = [1, 16, 128, 1024]
    epochs = [1, 10, 50, 50]
    model_list_losses = []
    model_list_accs = []
    output_dim = len(train_data['country'].unique())
    # train the network with batches and epochs according to the index
    # plot the val loss
    for batch_size, epochs in zip(batch_size, epochs):
        model = build_nn(2, output_dim, depth=6, width=16, batch_norm=False)
        model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses = train_model(
            train_data, val_data, test_data, model, lr=0.01, epochs=epochs, batch_size=batch_size)
        model_list_losses.append((batch_size, val_losses))
        model_list_accs.append((batch_size, val_accs))

    # Plot accuracies on the test set per model
    plt.figure()
    plt.title('Validation Accuracies')
    plt.xlabel('Batch Size')
    plt.ylabel('Accuracy')
    for (batch_size, val_accs) in model_list_accs:
        plt.plot(val_accs, label=f'Batch Size={batch_size}')
    plt.legend()
    plt.show()

    plt.figure()
    plt.title('Validation Losses')
    plt.xlabel('Batch Size')
    plt.ylabel('Loss')
    for (batch_size, val_losses) in model_list_losses:
        plt.plot(val_losses, label=f'Batch Size={batch_size}')
    plt.legend()
    plt.show()


def train_8_classifieres(train_data, test_data, val_data):
    depths = [1, 2, 6, 10, 6, 6, 6]
    widths = [16, 16, 16, 16, 8, 32, 64]
    best_model = None
    best_val_acc = 0
    train_accs = []
    val_accs = []
    test_accs = []
    train_losses = []
    val_losses = []
    test_losses = []
    best_training_loss = []
    best_val_loss = []
    best_test_loss = []

    for depth, width in zip(depths, widths):
        out_dim = len(train_data['country'].unique())
        model = build_nn(2, out_dim, depth, width, batch_norm=False)
        model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses = train_model(
            train_data, val_data, test_data, model, lr=0.01, epochs=50, batch_size=256)
        if val_accs[-1] > best_val_acc:
            best_val_acc = val_accs[-1]
            best_model = model
            best_training_loss = train_losses
            best_val_loss = val_losses
            best_test_loss = test_losses
    plt.figure()
    plt.plot(best_training_loss, label='Train', color='red')
    plt.plot(best_val_loss, label='Val', color='blue')
    plt.plot(best_test_loss, label='Test', color='green')
    plt.autumn()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Best model losses')
    plt.legend()
    plt.show()


def train_the_best_model(train_data, test_data, val_data):
    out_dim = len(train_data['country'].unique())
    model = build_nn(2, out_dim, depth=6, width=64, batch_norm=True)
    model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses = train_model(
        train_data, val_data, test_data, model, lr=0.01, epochs=50, batch_size=32)
    plt.figure()
    plt.plot(train_losses, label='Train', color='red')
    plt.plot(val_losses, label='Val', color='blue')
    plt.plot(test_losses, label='Test', color='green')
    plt.autumn()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Best model losses')
    plt.legend()
    plt.show()
    # TypeError: plot_decision_boundaries() missing 1 required positional argument: 'y'
    plot_decision_boundaries(
        model, test_data[['long', 'lat']].values, test_data['country'].values)


def train_the_worst_model(train_data, test_data, val_data):
    out_dim = len(train_data['country'].unique())
    model = build_nn(2, out_dim, depth=1, width=16, batch_norm=True)
    model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses = train_model(
        train_data, val_data, test_data, model, lr=0.001, epochs=50, batch_size=32)
    plt.figure()
    plt.plot(val_losses, label='Val', color='blue')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Worst model losses")
    plt.legend()
    plt.show()
    plot_decision_boundaries(
        model, test_data[['long', 'lat']].values, test_data['country'].values)

    plt.figure()
    plt.plot(val_accs, label='Val', color='blue')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Worst model accuracy")
    plt.legend()
    plt.show()


def train_MLP_width_16(train_data, test_data, val_data):
    """check the effect of the depth of the network on the performance"""
    out_dim = len(train_data['country'].unique())
    val_losses_of_all = []
    val_accs_of_all = []
    plt.figure()
    for depth in [1, 2, 6, 10]:
        model = build_nn(2, out_dim, depth, 16, batch_norm=True)
        model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses = train_model(
            train_data, val_data, test_data, model, lr=0.001, epochs=50, batch_size=32)
        val_losses_of_all.append(val_losses)
        val_accs_of_all.append(val_accs)

        plot_decision_boundaries(
            model, test_data[['long', 'lat']].values, test_data['country'].values)
    colors = iter(['red', 'blue', 'green', 'orange', 'purple', 'black'])
    for depth in enumerate([1, 2, 6, 10]):
        plt.plot(val_losses_of_all[depth[0]],
                 label=f'depth={depth[1]}', color=colors.__next__())
    plt.title('Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.figure()
    colors = iter(['red', 'blue', 'green', 'orange', 'purple', 'black'])
    for depth in enumerate([1, 2, 6, 10]):
        plt.plot(val_accs_of_all[depth[0]],
                 label=f'depth={depth[1]}', color=colors.__next__())
    plt.title('Validation Accuracies')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def train_MLP_depth_of_6(train_data, test_data, val_data):
    """check the effect of the width of the network on the performance"""
    out_dim = len(train_data['country'].unique())
    training_accs_of_all = []
    val_accs_of_all = []
    test_accs_of_all = []
    plt.figure()
    for width in [8, 32, 64]:
        model = build_nn(2, out_dim, 6, width, batch_norm=True)
        model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses = train_model(
            train_data, val_data, test_data, model, lr=0.001, epochs=30, batch_size=32)
        training_accs_of_all.append(train_accs)
        val_accs_of_all.append(val_accs)
        test_accs_of_all.append(test_accs)

    colors = iter(['red', 'blue', 'green', 'orange', 'purple', 'black'])
   # we plot the plot the training,
    #  validation and test accuracy of the models vs. number of neuron in each
    #  hidden layer
    plt.figure()
    for width in enumerate([8, 32, 64]):
        plt.plot(training_accs_of_all[width[0]],
                 label=f'width={width[1]}', color=colors.__next__())
    plt.title('Training Accuracies')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    plt.figure()

    for width in enumerate([8, 32, 64]):
        plt.plot(val_accs_of_all[width[0]],
                 label=f'width={width[1]}', color=colors.__next__())
    plt.title('Validation Accuracies')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    plt.figure()
    colors = iter(['red', 'blue', 'green', 'orange', 'purple', 'black'])
    for width in enumerate([8, 32, 64]):
        plt.plot(test_accs_of_all[width[0]],
                 label=f'width={width[1]}', color=colors.__next__())
    plt.title('Test Accuracies')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def train_model_with_gradients(train_data, val_data, test_data, model, lr=0.001, epochs=50, batch_size=256, layers_to_monitor=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print('Using device:', device)
    # load data
    trainset = torch.utils.data.TensorDataset(torch.tensor(
        train_data[['long', 'lat']].values).float(), torch.tensor(train_data['country'].values).long())
    valset = torch.utils.data.TensorDataset(torch.tensor(
        val_data[['long', 'lat']].values).float(), torch.tensor(val_data['country'].values).long())
    testset = torch.utils.data.TensorDataset(torch.tensor(
        test_data[['long', 'lat']].values).float(), torch.tensor(test_data['country'].values).long())

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=1024, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1024, shuffle=False, num_workers=0)
    # initialize lists to store the accuracies and losses

    train_accs = []
    val_accs = []
    test_accs = []
    train_losses = []
    val_losses = []
    test_losses = []
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss().to(device)
    grad_magnitudes = {layer: [] for layer in layers_to_monitor}
    for ep in range(epochs):
        model.train()
        preds_correct = 0
        ep_loss = 0
        # Reset ep_grads for each epoch
        ep_grads = {layer: [] for layer in layers_to_monitor}

        for i, (inputs, labels) in enumerate(tqdm(trainloader)):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            layer_idx = 0  # Reset layer index for each batch
            for _, layer in model.named_modules():
                if isinstance(layer, nn.Linear):
                    if layer_idx in layers_to_monitor:
                        grad_magnitude = torch.norm(
                            layer.weight.grad)**2 + torch.norm(layer.bias.grad)**2
                        ep_grads[layer_idx].append(grad_magnitude.item())
                    layer_idx += 1  # Increment layer index only once per linear layer

            optimizer.step()
            preds_correct += (torch.argmax(outputs, dim=1)
                              == labels).sum().item()
            ep_loss += loss.item()

        # Move these lines inside the epoch loop
        for layer in layers_to_monitor:
            grad_magnitudes[layer].append(np.mean(ep_grads[layer]))

        train_accs.append(preds_correct / len(trainset))
        train_losses.append(ep_loss / len(trainloader))
        # Validation and test evaluation should be inside the epoch loop as well

        model.eval()
        with torch.no_grad():
            for loader, accs, losses in zip([valloader, testloader], [val_accs, test_accs], [val_losses, test_losses]):
                correct = 0
                total = 0
                ep_loss = 0
                for inputs, labels in loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    ep_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                accs.append(correct / total)
                losses.append(ep_loss / len(loader))

    # Printing epoch-wise information should also be inside the epoch loop
    print('Epoch {:}, Train Acc: {:.3f}, Val Acc: {:.3f}, Test Acc: {:.3f}'.format(
        ep, train_accs[-1], val_accs[-1], test_accs[-1]))
    print('Grad Magnitudes:', grad_magnitudes)
    print('ep grads:', ep_grads)

    return model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, grad_magnitudes, ep_grads


def train_model(train_data, val_data, test_data, model, lr=0.001, epochs=50, batch_size=256, implicit_representation=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print('Using device:', device)

    if implicit_representation:
        alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        train_lon_lat_data = train_data[['long', 'lat']]
        val_lon_lat_data = val_data[['long', 'lat']]
        test_lon_lat_data = test_data[['long', 'lat']]

        train_sine_transformed_data = []
        val_sine_transformed_data = []
        test_sine_transformed_data = []

        # Apply the sine function to each element of 'long' and 'lat' columns multiplied by each alpha
        for alpha in alphas:
            train_sine_transformed_data.append(
                np.sin(train_lon_lat_data * alpha))
            val_sine_transformed_data.append(np.sin(val_lon_lat_data * alpha))
            test_sine_transformed_data.append(
                np.sin(test_lon_lat_data * alpha))

        # Concatenate results
        train_sine = np.concatenate(train_sine_transformed_data, axis=1)
        val_sine = np.concatenate(val_sine_transformed_data, axis=1)
        test_sine = np.concatenate(test_sine_transformed_data, axis=1)

        # Make tensors
        train_set = torch.utils.data.TensorDataset(torch.tensor(train_sine).float(),
                                                   torch.tensor(train_data['country'].values).long())
        val_set = torch.utils.data.TensorDataset(torch.tensor(val_sine).float(),
                                                 torch.tensor(val_data['country'].values).long())
        test_set = torch.utils.data.TensorDataset(torch.tensor(test_sine).float(),
                                                  torch.tensor(test_data['country'].values).long())

    else:
        train_set = torch.utils.data.TensorDataset(torch.tensor(train_data[['long', 'lat']].values).float(),
                                                   torch.tensor(train_data['country'].values).long())
        val_set = torch.utils.data.TensorDataset(torch.tensor(val_data[['long', 'lat']].values).float(),
                                                 torch.tensor(val_data['country'].values).long())
        test_set = torch.utils.data.TensorDataset(torch.tensor(test_data[['long', 'lat']].values).float(),
                                                  torch.tensor(test_data['country'].values).long())

    trainloader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(
        val_set, batch_size=1024, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(
        test_set, batch_size=1024, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_accs = []
    val_accs = []
    test_accs = []
    train_losses = []
    val_losses = []
    test_losses = []

    # Initialize lists to store gradient magnitudes for specific layers
    num_layers = [0, 30, 60, 90, 95, 99]
    gradients_magnitudes = {layer: [] for layer in num_layers}

    for ep in range(epochs):
        model.train()
        pred_correct = 0
        ep_loss = 0.

        epoch_gradients = {layer: [] for layer in num_layers}

        for i, (inputs, labels) in enumerate(tqdm(trainloader)):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            layer_index = 0
            for _, x in model.named_modules():
                if isinstance(x, nn.Linear):
                    if layer_index in num_layers:
                        curr_grad_magnitude = torch.norm(
                            x.weight.grad)*2 + torch.norm(x.bias.grad)*2
                        epoch_gradients[layer_index].append(
                            curr_grad_magnitude.item())
                    layer_index += 1
            optimizer.step()
            pred_correct += (torch.argmax(outputs, dim=1)
                             == labels).sum().item()
            ep_loss += loss.item()

        for layer in num_layers:
            if len(epoch_gradients[layer]) > 0:
                gradients_magnitudes[layer].append(
                    np.mean(epoch_gradients[layer]))

        train_accs.append(pred_correct / len(train_set))
        train_losses.append(ep_loss / len(trainloader))

        model.eval()
        with torch.no_grad():
            for loader, accs, losses in zip([valloader, testloader], [val_accs, test_accs], [val_losses, test_losses]):
                correct = 0
                total = 0
                ep_loss = 0.
                for inputs, labels in loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    ep_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                accs.append(correct / total)
                losses.append(ep_loss / len(loader))

        # val_loss_per_epoch.append(val_losses[-1])
        print('Epoch {:}, Train Acc: {:.3f}, Val Acc: {:.3f}, Test Acc: {:.3f}'.format(ep, train_accs[-1], val_accs[-1],
                                                                                       test_accs[-1]))

    return model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, gradients_magnitudes


def plot_gradients_magnitude(grad_magnitude, layers_to_monitor):
    colors = iter(['red', 'blue', 'green', 'orange', 'purple', 'black'])
    plt.figure()
    for layer in enumerate(layers_to_monitor):
        plt.plot(grad_magnitude[layer[1]],
                 label=f'Layer {layer[0]}', color=colors.__next__())
    plt.title('Gradient Magnitude vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Gradient Magnitude')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # seed for reproducibility
    torch.manual_seed(24)
    np.random.seed(24)

    train_data = pd.read_csv('train.csv')
    val_data = pd.read_csv('validation.csv')
    test_data = pd.read_csv('test.csv')

    # train_8_classifieres(train_data, test_data, val_data)
    # get_epochs_losses(train_data, val_data, test_data)
    # get_epochs_losses_no_batch_normal(train_data, val_data, test_data)
    # plot them against each other
    # plot_epochs_normal_vs_no_batch_normal(train_data, val_data, test_data)
    # question4(train_data, val_data, test_data)
    # train_the_best_model(train_data, test_data, val_data)
    # train_the_worst_model(train_data, test_data, val_data)
    # train_MLP_width_16(train_data, test_data, val_data)
    # # train_MLP_depth_of_6(train_data,test_data,val_data)
    # model_list = check_effect_of_batch_size_and_plot(train_data, val_data, test_data)
    # plot_batch_size_effect(model_list)
    # # monitor 0,30,60,90,95,99
    # out_dim = len(train_data['country'].unique())
    # # train model
    # model = build_nn(2, out_dim, depth=6, width=6, batch_norm=False)
    # model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, grad_magnitudes = train_model(train_data, val_data, test_data, model, lr=0.001, epochs=50, batch_size=256, implicit_representation=False)
    # plot_gradients_magnitude(grad_magnitudes, [0, 30, 60, 90, 95, 99])
    # use XGBOOST with defualt paramters
    