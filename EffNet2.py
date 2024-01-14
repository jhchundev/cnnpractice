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
            nn.SiLU(),
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
            layers = [ConvBN(ch_in, ch_med, 1), nn.SiLU()]
        else:
            layers = []

        layers.extend([
            ConvBN(ch_med, ch_med, kernel_size, stride=stride,
                   padding=(kernel_size - 1) // 2, groups=ch_med),  # depth-wise
            nn.SiLU(),
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
        features.extend([ConvBN(input_ch, ch_out, 3, stride=2), nn.SiLU()])

        ch_in = ch_out
        for t, c, n, s, k in settings:
            ch_out = int(math.ceil(c * width_mult))
            repeats = int(math.ceil(n * depth_mult))
            for i in range(repeats):
                stride = s if i == 0 else 1
                features.extend([BMConvBlock(ch_in, ch_out, t, stride, k)])
                ch_in = ch_out

        ch_last = int(math.ceil(1280 * width_mult))
        features.extend([ConvBN(ch_in, ch_last, 1), nn.SiLU()])

        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
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


def main():
    model = efficientnet_b7()

    # Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)

    # Training loop
    for epoch in range(5):
        # Validation loop
        model.eval()
        acc_list = []
        with torch.no_grad():
            for batch in tqdm(testloader):
                image, label = batch[0].to(device), batch[1].to(device)
                pred = model(image)
                accuracy = 100 * torch.sum(torch.argmax(pred, dim=1) == label) / len(pred)
                acc_list.append(accuracy.item())
    print(np.mean(acc_list))


def _efficientnet(w_mult, d_mult, resolution, drop_rate,
                  input_ch, num_classes=1000):
    model = EfficientNet(w_mult, d_mult,
                         resolution, drop_rate,
                         input_ch, num_classes)
    return model


def efficientnet_b7(input_ch=3, num_classes=1000):
    # (w_mult, d_mult, resolution, droprate) = (2.0, 3.1, 600, 0.5)
    return _efficientnet(2.0, 3.1, None, 0.5, input_ch, num_classes)


if __name__ == '__main__':
    main()
