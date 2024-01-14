import math
import optuna
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


class SEblock(nn.Module):  # Squeeze Excitation
    def __init__(self, ch_in, ch_sq):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch_in, ch_sq, 1),
            Swish(),
            nn.Conv2d(ch_sq, ch_in, 1),
        )
        self.se.apply(weights_init)

    def forward(self, x):
        return x * torch.sigmoid(self.se(x))


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)

    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)


class ConvBN(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size,
                 stride=1, padding=0, groups=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size,
                      stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(ch_out),
        )
        self.layers.apply(weights_init)

    def forward(self, x):
        return self.layers(x)


class Swish(nn.Module):  # Swish activation
    def forward(self, x):
        return x * torch.sigmoid(x)


class DropConnect(nn.Module):
    def __init__(self, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate

    def forward(self, x):
        if self.training:
            keep_rate = 1.0 - self.drop_rate
            r = torch.rand([x.size(0), 1, 1, 1], dtype=x.dtype).to(x.device)
            r += keep_rate
            mask = r.floor()
            return x.div(keep_rate) * mask
        else:
            return x


class BMConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out,
                 expand_ratio, stride, kernel_size,
                 reduction_ratio=4, drop_connect_rate=0.2):
        super().__init__()
        self.use_residual = (ch_in == ch_out) & (stride == 1)
        ch_med = int(ch_in * expand_ratio)
        ch_sq = max(1, ch_in // reduction_ratio)

        # define network
        if expand_ratio != 1.0:
            layers = [ConvBN(ch_in, ch_med, 1), Swish()]
        else:
            layers = []

        layers.extend([
            ConvBN(ch_med, ch_med, kernel_size, stride=stride,
                   padding=(kernel_size - 1) // 2, groups=ch_med),  # depth-wise
            Swish(),
            SEblock(ch_med, ch_sq),  # Squeeze Excitation
            ConvBN(ch_med, ch_out, 1),  # pixel-wise
        ])

        if self.use_residual:
            self.drop_connect = DropConnect(drop_connect_rate)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.drop_connect(self.layers(x))
        else:
            return self.layers(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class EfficientNet(nn.Module):
    def __init__(self, width_mult=1.0, depth_mult=1.0,
                 resolution=False, dropout_rate=0.2,
                 input_ch=3, num_classes=1000):
        super().__init__()

        # expand_ratio, channel, repeats, stride, kernel_size
        settings = [
            [1, 16, 1, 1, 3],  # MBConv1_3x3, SE, 112 -> 112
            [6, 24, 2, 2, 3],  # MBConv6_3x3, SE, 112 ->  56
            [6, 40, 2, 2, 5],  # MBConv6_5x5, SE,  56 ->  28
            [6, 80, 3, 2, 3],  # MBConv6_3x3, SE,  28 ->  14
            [6, 112, 3, 1, 5],  # MBConv6_5x5, SE,  14 ->  14
            [6, 192, 4, 2, 5],  # MBConv6_5x5, SE,  14 ->   7
            [6, 320, 1, 1, 3]  # MBConv6_3x3, SE,   7 ->   7]
        ]

        ch_out = int(math.ceil(32 * width_mult))
        features = [nn.AdaptiveAvgPool2d(resolution)] if resolution else []
        features.extend([ConvBN(input_ch, ch_out, 3, stride=2), Swish()])

        ch_in = ch_out
        for t, c, n, s, k in settings:
            ch_out = int(math.ceil(c * width_mult))
            repeats = int(math.ceil(n * depth_mult))
            for i in range(repeats):
                stride = s if i == 0 else 1
                features.extend([BMConvBlock(ch_in, ch_out, t, stride, k)])
                ch_in = ch_out

        ch_last = int(math.ceil(1280 * width_mult))
        features.extend([ConvBN(ch_in, ch_last, 1), Swish()])

        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(ch_last, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


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
    # EfficientNet specific hyperparameters
    width_mult = trial.suggest_float('width_mult', 0.5, 2.0)
    depth_mult = trial.suggest_float('depth_mult', 0.5, 2.0)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)

    # Other hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    beta1 = trial.suggest_float("beta1", 0.9, 0.999)
    beta2 = trial.suggest_float("beta2", 0.9, 0.999)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    # Initialize the EfficientNet model
    model = EfficientNet(width_mult=width_mult, depth_mult=depth_mult, dropout_rate=dropout_rate)
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
