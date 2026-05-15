from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms



_CFG_VGG16 = [
    64, 64, 'M',
    128, 128, 'M',
    256, 256, 256, 'M',
    512, 512, 512, 'M',
    512, 512, 512, 'M'
]


def _make_layers_vgg16(cfg, batch_norm=True):
    layers = []
    in_channels = 3

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(
                in_channels, v, kernel_size=3, padding=1,
                bias=not batch_norm
            )

            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]

            in_channels = v

    return nn.Sequential(*layers)


class VGG16(nn.Module):
    def __init__(
        self,
        num_classes=10,
        batch_norm=True,
        classifier_hidden=512,
        dropout_p=0.5,
    ):
        super().__init__()

        self.features = _make_layers_vgg16(_CFG_VGG16, batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if classifier_hidden > 0:
            self.classifier = nn.Sequential(
                nn.Linear(512, classifier_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_p),
                nn.Linear(classifier_hidden, num_classes),
            )
        else:
            self.classifier = nn.Linear(512, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_model(
    num_classes=10,
    batch_norm=True,
    classifier_hidden=512,
    dropout_p=0.5,
):
    return VGG16(
        num_classes=num_classes,
        batch_norm=batch_norm,
        classifier_hidden=classifier_hidden,
        dropout_p=dropout_p,
    )


# ---------------------------------------------------------------------------
# Training utilities (UNCHANGED)
# ---------------------------------------------------------------------------

def create_dataloaders(data_dir, batch_size, num_workers, cutout=0):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    train_transforms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    if cutout > 0:
        train_transforms.append(
            transforms.RandomErasing(p=0.5, scale=(cutout / 32, cutout / 32), ratio=(1, 1))
        )

    train_tf = transforms.Compose(train_transforms)

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
    test_set  = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0
    criterion = nn.CrossEntropyLoss()

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)

        loss_sum += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == targets).sum().item()
        total += images.size(0)

    return loss_sum / total, 100.0 * correct / total


def train(model, epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader, val_loader = create_dataloaders("./data", 128, 4)

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.5)

    for epoch in range(epochs):
        model.train()
        total, correct, loss_sum = 0, 0, 0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = nn.CrossEntropyLoss()(outputs, targets)

            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()
            total += images.size(0)

        scheduler.step()

        val_loss, val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}: "
              f"Train Acc {100*correct/total:.2f}% | "
              f"Val Acc {val_acc:.2f}%")
