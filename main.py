import optuna
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from cnn2 import CNN2

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
