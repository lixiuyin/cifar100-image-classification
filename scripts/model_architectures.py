import os
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["WRN_DEPTH"] = "28"
os.environ["WRN_WIDTH"] = "10"
os.environ["WRN_DROPOUT"] = "0.3"

class BasicBlock(nn.Module):
    """
    Pre-activation WideResNet Basic Block.
    Follows the original WideResNet paper implementation.
    """
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    """Stack of BasicBlocks forming a network block."""
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, drop_rate)
    
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, drop_rate))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    """
    WideResNet implementation for CIFAR-10/100.
    
    Args:
        depth (int): Network depth (e.g., 28, 40). Must satisfy (depth-4) % 6 == 0
        num_classes (int): Number of output classes
        widen_factor (int): Width multiplier (default: 10)
        drop_rate (float): Dropout rate (default: 0.3)
    """
    def __init__(self, depth, num_classes, widen_factor=10, drop_rate=0.3):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, drop_rate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        # Use adaptive pooling to support arbitrary input sizes
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return self.fc(out)
    
    def _initialize_weights(self):
        """Initialize model weights using proper schemes."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming initialization for Conv layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # BN layers: weight=1, bias=0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Linear layer: small normal distribution
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def create_model(num_classes, device):
    """
    Create WideResNet-28-10 model for CIFAR classification.
    
    Args:
        num_classes (int): Number of output classes (10 for CIFAR-10, 100 for CIFAR-100)
        device (str or torch.device): Device to place the model on
    
    Returns:
        WideResNet model on specified device
    """
    model = WideResNet(depth=int(os.environ.get("WRN_DEPTH", 28)),
                       num_classes=num_classes,
                       widen_factor=int(os.environ.get("WRN_WIDTH", 10)),
                       drop_rate=float(os.environ.get("WRN_DROPOUT", 0.3)))
    model = model.to(device)
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🏗️ Model: WideResNet-{os.environ.get('WRN_DEPTH', 28)}-{os.environ.get('WRN_WIDTH', 10)}")
    print(f"   Total parameters: {num_params:,}")
    print(f"   Trainable parameters: {num_trainable:,}")
    
    return model