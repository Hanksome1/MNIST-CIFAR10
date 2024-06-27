# https://pytorch.org/tutorials/beginner/basics/intro.html

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,32,kernel_size=3, padding = 1)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3, padding = 1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(p=0.8)

    def forward(self, x):
        # TODO: Define the forward pass of the model
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


def train(args, model, train_loader, optimizer, epoch):
    # Set the model to training mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_function = nn.CrossEntropyLoss()
    model.train()
    loss = 0
    # TODO: Define the training loop
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        training_loss = loss_function(outputs, target)
        training_loss.backward()
        optimizer.step()
        loss+=training_loss.item()
    average_loss = loss/len(train_loader)
    print("Epoch:", epoch, " loss: " , average_loss)
    return average_loss

def test(model, test_loader):
    # Set the model to evaluation mode
    model.eval()
    test_loss = 0
    total = 0
    correct = 0
    loss = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_function = nn.CrossEntropyLoss()
    # TODO: Define the testing loop
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device) , target.to(device)
            outputs = model(data)
            testing_loss = loss_function(outputs, target)
            _ , predicted = torch.max(outputs.data,1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            loss+= testing_loss.item()
    average_loss = loss/len(test_loader)
    print("Testing Loss : ", average_loss)
    accuracy = 100* correct/total
    print("Accuracy :" , accuracy )
    return average_loss
    # Log the testing status


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.5, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.9, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # TODO: Tune the batch size to see different results
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    # Set transformation for the dataset
    # TODO: (Bonus) Change different dataset and transformations (data augmentation)
    # https://pytorch.org/vision/stable/datasets.html
    # https://pytorch.org/vision/main/transforms.html
    # e.g. CIFAR-10, Caltech101, etc. 
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    model = Net().to(device)
    # TODO: (Bonus) Tune the learning rate / optimizer to see different results
    weight_decay = 3e-3
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    # TODO: (Bonus) Tune the learning rate scheduler to see different results
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    testing_loss_list = []
    training_loss_list = []
    for epoch in range(1, args.epochs + 1):
        # TODO: (Bonus) Return the loss and accuracy of the training loop and plot them
        # https://matplotlib.org/stable/tutorials/pyplot.html
        training_loss_list.append(train(args, model, train_loader, optimizer, epoch))
        testing_loss_list.append(test(model, test_loader))
        scheduler.step()
    plt.plot(training_loss_list)
    plt.plot(testing_loss_list)
    plt.legend(["training loss", "testing loss"])
    plt.savefig("loss.png")

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
