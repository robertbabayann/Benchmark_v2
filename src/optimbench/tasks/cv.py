import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import timm


from optimbench.tasks import Task


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, 1, 1, bias=False)
        self.drop_rate = drop_rate
        self.equal_io = in_planes == out_planes
        self.shortcut = None if self.equal_io else nn.Conv2d(in_planes, out_planes, 1, stride, 0, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = x if self.equal_io else self.shortcut(out)
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return out + shortcut


class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, num_classes=100, drop_rate=0.0):
        super().__init__()
        n = (depth - 4) // 6
        widths = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        self.conv1 = nn.Conv2d(3, widths[0], 3, 1, 1, bias=False)
        self.block1 = self._make_layer(widths[0], widths[1], n, 1, drop_rate)
        self.block2 = self._make_layer(widths[1], widths[2], n, 2, drop_rate)
        self.block3 = self._make_layer(widths[2], widths[3], n, 2, drop_rate)
        self.bn = nn.BatchNorm2d(widths[3])
        self.fc = nn.Linear(widths[3], num_classes)

    def _make_layer(self, in_planes, out_planes, n, stride, drop_rate):
        layers = [BasicBlock(in_planes, out_planes, stride, drop_rate)]
        for _ in range(n - 1):
            layers.append(BasicBlock(out_planes, out_planes, 1, drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = F.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


class Cifar100Task(Task):
    name = "cifar100_wrn28_10"
    higher_is_better = True
    max_steps_cap = 20000
    checkpoint_every = 200

    def __init__(self, data_root, batch_size=128):
        self.data_root = data_root
        self.batch_size = batch_size

    def build_model(self, seed):
        torch.manual_seed(seed)
        return WideResNet()

    def build_data(self, seed):
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        val_tf = transforms.ToTensor()
        train_set = torchvision.datasets.CIFAR100(self.data_root, train=True, download=True, transform=train_tf)
        val_set = torchvision.datasets.CIFAR100(self.data_root, train=False, download=True, transform=val_tf)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=2)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=256, shuffle=False, num_workers=2)
        return train_loader, val_loader

    def loss_and_metric(self, model, batch):
        x, y = batch
        return F.cross_entropy(model(x), y)

    def evaluate(self, model, val_loader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        model.train()
        return correct / total


class TinyImageNetTask(Task):
    name = "tinyimagenet_swin_tiny"
    higher_is_better = True
    max_steps_cap = 20000
    checkpoint_every = 200

    def __init__(self, data_root, batch_size=64):
        self.data_root = data_root
        self.batch_size = batch_size

    def build_model(self, seed):
        torch.manual_seed(seed)
        return timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=200)

    def build_data(self, seed):
        tf = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
        train_set = torchvision.datasets.ImageFolder(f"{self.data_root}/train", transform=tf)
        val_set = torchvision.datasets.ImageFolder(f"{self.data_root}/val", transform=tf)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=2)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=128, shuffle=False, num_workers=2)
        return train_loader, val_loader

    def loss_and_metric(self, model, batch):
        x, y = batch
        return F.cross_entropy(model(x), y)

    def evaluate(self, model, val_loader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        model.train()
        return correct / total
