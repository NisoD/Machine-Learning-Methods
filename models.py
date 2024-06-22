import numpy as np
import torch
from helpers import read_data_demo, plot_decision_boundaries
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, TensorDataset, DataLoader
import pandas as pd
from torch import optim
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree

class Ridge_Regression:

    def __init__(self, lambd):
        self.lambd = lambd

    def fit(self, X, Y):
        """
        Fit the ridge regression model to the provided data.
        :param X: The training features.
        :param Y: The training labels.
        """
        Y = 2 * (Y - 0.5)  # make it 1 , -1 to make it compatible with the formula
        N = X.shape[0]
        X_T = X.transpose()
        X_Mult = X_T.dot(X)  # Matrix Multiplication
        X_Mult /= N
        self.w = np.linalg.inv(X_Mult + self.lambd *
                               np.identity(X.shape[1])).dot(X_T).dot(Y) / N
        # transform the labels to -1 and 1, instead of 0 and 1.

        ########## YOUR CODE HERE ##########
        # Fit is the train function
        # compute the ridge regression weights using the formula from class / exercise.
        # you may not use np.linalg.solve, but you may use np.linalg.inv

        ####################################
        pass

    def predict(self, X):
        """
        Predict the output for the provided data.
        :param X: The data to predict. np.ndarray of shape (N, D).
        :return: The predicted output. np.ndarray of shape (N,), of 0s and 1s.
        """
        np.sign
        # np sign returns 1 if positive -1 if negative
        preds = np.sign(X.dot(self.w))
        preds = (preds + 1) / 2  # make it 0,1 as was in data
        return preds

        ########## YOUR CODE HERE ##########

        # compute the predicted output of the model.
        # name your predicitons array preds.

        ####################################

        # transform the labels to 0s and 1s, instead of -1s and 1s.
        # You may remove this line if your code already outputs 0s and 1s.


class Logistic_Regression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Logistic_Regression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        Computes the output of the linear operator.
        :param x: The input to the linear operator.
        :return: The transformed input.
        """
       
        out = self.linear(x)
        return out
   
    def predict(self, x):
        """
        THIS FUNCTION IS NOT NEEDED FOR PYTORCH. JUST FOR OUR VISUALIZATION
        """
        x = torch.from_numpy(x).float().to(self.linear.weight.data.device)
        x = self.forward(x)
        x = nn.functional.softmax(x, dim=1)
        x = x.detach().cpu().numpy()
        x = np.argmax(x, axis=1)
        return x


def scenario1():
    data, _ = read_data_demo('train.csv')
    test_data, _ = read_data_demo('test.csv')

    lambda_values = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    accuracies = []
    for lambda_val in lambda_values:
        model = Ridge_Regression(lambda_val)
        model.fit(data[:, :-1], data[:, -1])  # data and labels accordingly
        preds = model.predict(test_data[:, :-1])  # data to predict
        # acc. of the test and predcition
        accuracies.append(np.mean(preds == test_data[:, -1]))
        # plot the data
    plt.plot(lambda_values, accuracies, marker='o')
    plt.xlabel('Lambda')
    plt.ylabel('Accuracy')
    plt.title('Ridge Regression')
    plt.show()


def gradientDescent():
    """ This function has to calculate gradient descent using numpy
    f(x,y)=(x-3)^2 +(y-5)^2
    optimize (x,y) with learning rate of 0.1
    do 1000 iterations
    init (x,y) = (0,0) 
    Plot your optimized vector through the iterations (x axis- x, y axis- y).
 Color the points by the “time” (iterations).
    """
    x, y = 0, 0
    learnig_rate = 0.1
    iterations = 1000
    x_values = []
    y_values = []
    for i in range(iterations):
        x -= learnig_rate * 2 * (x - 3)
        y -= learnig_rate * 2 * (y - 5)
        x_values.append(x)
        y_values.append(y)
    print("Last value of x:", x)
    print("Last value of y:", y)
    plt.scatter(x_values, y_values, c=range(iterations), marker='o')
    plt.colorbar(label='Iterations')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gradient Descent')
    plt.show()


def dataLoader():
    train_set = read_data_demo('train.csv')
    test_set = read_data_demo('test.csv')
    validation_set = read_data_demo('validation.csv')
    X_train, Y_train = train_set[0][:, :2], train_set[0][:, 2]
    X_test, Y_test = test_set[0][:, :2], test_set[0][:, 2]
    X_val, Y_val = validation_set[0][:, :2], validation_set[0][:, 2]
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.long)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.long)

    X_val = torch.tensor(X_val, dtype=torch.float32)
    Y_val = torch.tensor(Y_val, dtype=torch.long)
    # load as dataloader
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = TensorDataset(X_test, Y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    val_dataset = TensorDataset(X_val, Y_val)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    return train_loader, test_loader, val_loader


def train_model(learning_rate):
    train_loader,test_loader,val_loader=dataLoader() #load the data
    #get model 
    model = Logistic_Regression(2, 2)
    #set the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.3)
    num_epochs = 10
    ep_loss_values = []
    acc_values = []   
    train_accuracy = []
    val_accuracy = []
    test_accuracy = []
    train_losses = []
    ep_test_losses = []
    ep_val_losses = []

    for epoch in range(num_epochs): #how many times we want to update the weights on the dataest
        loss_values = []
        ep_correct_preds = 0.
        model.train()  # set the model to training mode
        for inputs, labels in train_loader:
            optimizer.zero_grad() # zero the parameter gradients each time
            outputs = model(inputs) # forward pass
            loss = criterion(outputs.squeeze(), labels) # calculate the loss with the loss we chose
            loss.backward() # backward pass backpropagation 
            optimizer.step() # update the weights

            # Store the loss values for plotting
            loss_values.append(loss.item())
            ep_correct_preds += torch.sum(torch.argmax(outputs, dim=1) == labels).item() #how many correct predictions from tensor the nubmer
        lr_scheduler.step()

        mean_loss = np.mean(loss_values)
        ep_accuracy = ep_correct_preds / len(train_loader.dataset)
        ep_loss_values.append(mean_loss)
        train_accuracy.append(ep_accuracy)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {mean_loss:.4f}, Accuracy: {ep_accuracy:.2f}')
        # end of every epoch, iterate over the validation and test sets and keep their losses accuracies
        model.eval()  # set the model to evaluation mode
        correct_predictions = 0.
        #validatiaon
        val_losses=[]
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                val_losses.append(loss.item())
                correct_predictions += torch.sum(torch.argmax(outputs, dim=1) == labels).item()
            val_accuracy.append(correct_predictions / len(val_loader.dataset))
            ep_val_losses.append(np.mean(val_losses))
        #test
        correct_predictions = 0.
        test_losses = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                test_losses.append(loss.item())
                correct_predictions += torch.sum(torch.argmax(outputs, dim=1) == labels).item()
            test_accuracy.append(correct_predictions / len(test_loader.dataset)
        )
        ep_test_losses.append(np.mean(test_losses))

    #plot the loss a
    plt.plot(ep_loss_values, label='Training Loss')
    plt.plot(ep_val_losses, label='Validation Loss')
    plt.plot(ep_test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()



    #in order to work with the plot function we need to convert the tensors to numpy
    train_data = train_loader.dataset.tensors[0].numpy()
    train_labels = train_loader.dataset.tensors[1].numpy()
    # and give it to the function
    title = f'Decision Boundary of learning rate {learning_rate}'
 
    plot_decision_boundaries(model, train_data, train_labels, title)  
    print(f'Test Accuracy: {correct_predictions / len(test_loader.dataset):.2f}, Learning Rate: {learning_rate}')  
    plt.show()
    
def multiple_class_loader():
    train_set = read_data_demo('train_multiclass.csv')
    test_set = read_data_demo('test_multiclass.csv')
    validation_set = read_data_demo('validation_multiclass.csv')
    X_train, Y_train = train_set[0][:, :2], train_set[0][:, 2]
    X_test, Y_test = test_set[0][:, :2], test_set[0][:, 2]
    X_val, Y_val = validation_set[0][:, :2], validation_set[0][:, 2]
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.long)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.long)

    X_val = torch.tensor(X_val, dtype=torch.float32)
    Y_val = torch.tensor(Y_val, dtype=torch.long)
    # load as dataloader
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = TensorDataset(X_test, Y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    val_dataset = TensorDataset(X_val, Y_val)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    return train_loader, test_loader, val_loader
def multiple_class_train_model(learning_rate):
    """ Train logistic regression classifiers on the train data (train multiclass.csv), with
 the following choices of initial learning rates: learning rate: 0.01,0.001,0.0003).
 5
Train your classifier for 30 epochs, with a batch size of 32. Decay the learning
 rate by 0.3 every 5 epochs (as seen in the tutorial).
 Compute the training, validation and test set losses and accuracies, for every
 epoch as done in Sec. 9.3
 # Define a learning rate scheduler (OPTIONAL)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                               step_size=15, # rate of decay
                                               gamma=0.3 # how much to decay the learning rate
                                              )
"""
    num_epochs = 30
    train_loader, test_loader, val_loader = multiple_class_loader()
    df = pd.read_csv('train_multiclass.csv')
    """Use the sklearn library to train a decision tree on the data
 (sklearn.tree.DecisionTreeClassifier). Use max
 depth = 2. Report
 the tree accuracy and visualize its predictions as before. """
   
    classes = df['country'].unique()
    model = Logistic_Regression(2, len(classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)
    ep_loss_values = []
    ep_test_losses = []
    ep_val_losses = []
    train_accuracy = []
    val_accuracy = []
    test_accuracy = []
    train_losses = []

    for epoch in range(num_epochs):
        loss_values = []
        ep_correct_preds = 0.
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            loss_values.append(loss.item()) 
            ep_correct_preds += torch.sum(torch.argmax(outputs, dim=1) == labels).item()

        lr_scheduler.step()
        mean_loss = np.mean(loss_values)
        ep_accuracy = ep_correct_preds / len(train_loader.dataset)
        ep_loss_values.append(mean_loss)
        train_accuracy.append(ep_accuracy)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {mean_loss:.4f}, Accuracy: {ep_accuracy:.2f}')
        
        model.eval()
        correct_predictions = 0.
        val_losses = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                val_losses.append(loss.item())
                correct_predictions += torch.sum(torch.argmax(outputs, dim=1) == labels).item()
            val_accuracy.append(correct_predictions / len(val_loader.dataset))
            ep_val_losses.append(np.mean(val_losses))
            print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {np.mean(val_losses):.4f}, Validation Accuracy: {correct_predictions / len(val_loader.dataset):.2f}')

        correct_predictions = 0.
        test_losses = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                test_losses.append(loss.item())
                correct_predictions += torch.sum(torch.argmax(outputs, dim=1) == labels).item()
            test_accuracy.append(correct_predictions / len(test_loader.dataset))
            ep_test_losses.append(np.mean(test_losses))
            print(f'Epoch [{epoch + 1}/{num_epochs}], Test Loss: {np.mean(test_losses):.4f}, Test Accuracy: {correct_predictions / len(test_loader.dataset):.2f}')  
    #plot the accuracies of the model on train,test,validation 
    plt.plot(test_accuracy, label='Test Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.plot(train_accuracy, label='Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    train_data = train_loader.dataset.tensors[0].numpy()
    train_labels = train_loader.dataset.tensors[1].numpy()
    title = f'Decision Boundary of learning rate {learning_rate}'
    plot_decision_boundaries(model, train_data, train_labels, title)
    print(f'Test Accuracy: {correct_predictions / len(test_loader.dataset):.2f}, Learning Rate: {learning_rate}')
    plt.show()  
def train_tree():
    train_data, _ = read_data_demo('train_multiclass.csv')
    test_data, _ = read_data_demo('test_multiclass.csv')
    val_data, _ = read_data_demo('validation_multiclass.csv')
    X_train = train_data[:, :2]
    X_test = test_data[:, :2]
    X_val = val_data[:, :2]
    Y_train = train_data[:, -1]
    Y_test = test_data[:, -1]
    Y_val = val_data[:, -1]
    
    tree = DecisionTreeClassifier(max_depth=2)

    tree.fit(X_train,Y_train)
    # test and validate
    test_accuracy = accuracy_score(Y_test,tree.predict(X_test))
    val_accuracy = accuracy_score(Y_val,tree.predict(X_val))
    print("Accuracy of the tree on test set:",test_accuracy*100,"%")
    print("Accuracy of the tree on validation set:",val_accuracy*100,"%")
    plot_decision_boundaries(tree, X_train, Y_train, title='Decision Boundaries of Decision Tree')
def train_bigger_tree():
    train_data, _ = read_data_demo('train_multiclass.csv')
    test_data, _ = read_data_demo('test_multiclass.csv')
    val_data, _ = read_data_demo('validation_multiclass.csv')
    X_train = train_data[:, :2]
    X_test = test_data[:, :2]
    X_val = val_data[:, :2]
    Y_train = train_data[:, -1]
    Y_test = test_data[:, -1]
    Y_val = val_data[:, -1]
    
    tree = DecisionTreeClassifier(max_depth=10)

    tree.fit(X_train,Y_train)
    # test and validate
    test_accuracy = accuracy_score(Y_test,tree.predict(X_test))
    val_accuracy = accuracy_score(Y_val,tree.predict(X_val))
    print("Accuracy of the tree on test set:",test_accuracy*100,"%")
    print("Accuracy of the tree on validation set:",val_accuracy*100,"%")
    plot_decision_boundaries(tree, X_train, Y_train, title='Decision Boundaries of Decision Tree')

def sceanrio5():
    train_model(0.1)
    train_model(0.01)
    train_model(0.001)
def scenario6():
    multiple_class_train_model(0.01)
    # multiple_class_train_model(0.001)
    # multiple_class_train_model(0.0003)
def plot_q1():
    #i didnt want to run all the epochs again 
    #just plotting
    learning_rates=[0.01,0.001,0.003]
    Accuracies=[0.84,0.79,0.79]
    plt.xlabel('Learning rate')
    plt.ylabel('Accuracy')
    plt.scatter(learning_rates,Accuracies,marker='o',linestyle='',color='blue')
    plt.show()

def main():
    # scneario1()
    # sceanrio5()
    # gradientDescent()
    # scenario6()
    # train_tree()
    train_bigger_tree()    
    # plot_q1()



if __name__ == '__main__':
    main()
