from torch.nn.functional import relu, softmax
import optuna
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


class CNN2(nn.Module):
    def __init__(self, conv_channels, linear_sizes, dropout_rate):
        super(CNN2, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, conv_channels[0], kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout laye  r
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layers
        self.fc1 = nn.Linear(conv_channels[2] * 4 * 4, linear_sizes[0])
        self.fc2 = nn.Linear(linear_sizes[0], linear_sizes[1])
        self.fc3 = nn.Linear(linear_sizes[1], 10)

    def forward(self, x):
        x = self.pool1(relu(self.conv1(x)))
        x = self.pool2(relu(self.conv2(x)))
        x = self.pool3(relu(self.conv3(x)))
        x = self.dropout(x)

        x = x.view(-1, self.num_flat_features(x))
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# データの前処理
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CIFAR-10 のダウンロード、読み込み
trainset = torchvision.datasets.CIFAR10(root='./data/', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data/', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)


def objective(trial):
    # Define hyperparameters
    conv_channels = [
        trial.suggest_int('conv1_channels', 8, 64),
        trial.suggest_int('conv2_channels', 16, 128),
        trial.suggest_int('conv3_channels', 32, 256)
    ]
    linear_sizes = [
        trial.suggest_int('fc1_size', 64, 512),
        trial.suggest_int('fc2_size', 32, 256)
    ]

    # Find HyperParameter to use Optuna
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    beta1 = trial.suggest_float("beta1", 0.9, 0.999)
    beta2 = trial.suggest_float("beta2", 0.9, 0.999)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)

    # Define model, optimizer, loss function
    model = CNN2(conv_channels, linear_sizes, dropout_rate)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    loss_func = nn.CrossEntropyLoss()

    # Check if GPU is available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # Training loop
    for epoch in range(5):
        model.train()
        for batch in tqdm(trainloader):
            image, label = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            pred = model(image)
            loss = loss_func(pred, label)
            loss.backward()
            optimizer.step()

        # Validation loop
        model.eval()
        acc_list = []
        with torch.no_grad():
            for batch in tqdm(testloader):
                image, label = batch[0].to(device), batch[1].to(device)
                pred = model(image)
                accuracy = 100 * torch.sum(torch.argmax(pred, dim=1) == label) / len(pred)
                acc_list.append(accuracy.item())

    return np.mean(acc_list)


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

best_params = study.best_params
print("Best parameters:", best_params)
