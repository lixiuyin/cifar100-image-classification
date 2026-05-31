import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda as cuda
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
from torch.amp import GradScaler, autocast

from tqdm import tqdm
import os
import math
import random
import numpy as np
from copy import deepcopy
from typing import Tuple, Optional, Dict, Any, List

os.environ["WARMUP_EPOCHS"] = "5"
os.environ["TOTAL_EPOCHS"] = "30"
os.environ["WARMUP_START_LR"] = "1e-6"
os.environ["FINAL_LR"] = "1e-6"
os.environ["CLIP_NORM"] = "1.0"
os.environ["MIXUP_ALPHA"] = "0.4"
os.environ["CUTMIX_ALPHA"] = "1.0"
os.environ["MIX_PROB"] = "0.5"
os.environ["NUM_WORKERS"] = "4"
os.environ["USE_AMP"] = "1"
os.environ["USE_MIXUP"] = "1"
os.environ["USE_CUTMIX"] = "1"
os.environ["USE_ONLINE_AUGMENTATION"] = "1"
os.environ["RANDOM_ERASING_PROB"] = "0.5"
os.environ["LABEL_SMOOTHING"] = "0.1"
os.environ["EMA_DECAY"] = "0.999"

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy Loss with Label Smoothing
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        
    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class WarmupCosineScheduler:
    """
    Standard Warmup + Cosine Annealing learning rate scheduler.
    
    This scheduler implements:
    1. Linear warmup from warmup_start_lr to base_lr over warmup_epochs
    2. Cosine annealing from base_lr to final_lr over remaining epochs
    
    This is an epoch-based scheduler. Call step() once per epoch (not per batch).

    Environment variables:
      - WARMUP_EPOCHS (default 5)
      - TOTAL_EPOCHS (default 30) 
      - WARMUP_START_LR (default 1e-6) - Starting lr for warmup phase
      - FINAL_LR (default 1e-6) - Final lr after cosine annealing
    
    Usage:
        scheduler = WarmupCosineScheduler(optimizer)
        for epoch in range(total_epochs):
            train(...)
            validate(...)
            scheduler.step()  # Call once at the end of each epoch
    """

    def __init__(self, optimizer: optim.Optimizer, base_lrs: List[float] = None):
        """
        Initialize the scheduler.
        
        Args:
            optimizer: PyTorch optimizer instance
            base_lrs: List of base learning rates for each param group.
                     If None, uses current lr from optimizer.
        """
        self.optimizer = optimizer
        self.warmup_epochs = int(os.environ.get("WARMUP_EPOCHS", 5))
        self.total_epochs = int(os.environ.get("TOTAL_EPOCHS", 30))
        self.warmup_start_lr = float(os.environ.get("WARMUP_START_LR", 1e-6))
        self.final_lr = float(os.environ.get("FINAL_LR", 1e-6))
        self.last_epoch = 0  # Current epoch, will be incremented by step()
        
        # Store base learning rates
        if base_lrs is None:
            self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        else:
            if len(base_lrs) != len(optimizer.param_groups):
                raise ValueError(f"Expected {len(optimizer.param_groups)} base_lrs, got {len(base_lrs)}")
            self.base_lrs = list(base_lrs)
        
        # Validate configuration
        if self.warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {self.warmup_epochs}")
        if self.total_epochs <= 0:
            raise ValueError(f"total_epochs must be > 0, got {self.total_epochs}")
        if self.warmup_epochs >= self.total_epochs:
            raise ValueError(f"warmup_epochs ({self.warmup_epochs}) must be < total_epochs ({self.total_epochs})")
        
        # Set initial learning rate (epoch 0)
        self._set_lr(0)

    def step(self):
        """
        Update learning rate for the next epoch.
        
        Call this method once at the end of each epoch to update the learning rate.
        """
        self.last_epoch += 1
        self._set_lr(self.last_epoch)

    def _set_lr(self, epoch: int):
        """Set learning rate for all parameter groups based on current epoch."""
        for i, group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i]
            lr = self._compute_lr(epoch, base_lr)
            group['lr'] = lr

    def get_last_lr(self) -> List[float]:
        """
        Return current learning rates for all parameter groups.
        
        Returns:
            List of current learning rates
        """
        return [group['lr'] for group in self.optimizer.param_groups]

    def _compute_lr(self, epoch: int, base_lr: float) -> float:
        """
        Compute learning rate for given epoch and base_lr.
        
        Phase 1 (Warmup): Linear increase from warmup_start_lr to base_lr
        Phase 2 (Cosine): Cosine annealing from base_lr to final_lr
        
        Args:
            epoch: Current epoch number (0-indexed)
            base_lr: Base learning rate for this param group
            
        Returns:
            Computed learning rate
        """

        if epoch >= self.total_epochs:
            return self.final_lr
        
        # Phase 1: Linear warmup from warmup_start_lr to base_lr
        if epoch < self.warmup_epochs:
            # Linear interpolation: epoch 0 → warmup_start_lr
            # Note: Last warmup epoch approaches but doesn't reach base_lr
            # The first cosine epoch (epoch == warmup_epochs) starts exactly at base_lr
            alpha = epoch / self.warmup_epochs
            return self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha
        
        # Phase 2: Cosine annealing from base_lr to final_lr
        cosine_epochs = self.total_epochs - self.warmup_epochs
        
        # Handle edge case: only 1 epoch in cosine phase
        if cosine_epochs <= 1:
            return self.final_lr
        
        # Calculate progress through cosine phase [0, 1]
        # When epoch = warmup_epochs, progress = 0
        # When epoch = total_epochs - 1, progress = 1
        progress = (epoch - self.warmup_epochs) / (cosine_epochs - 1)
        progress = max(0.0, min(1.0, progress))  # Clamp to [0, 1]
        
        # Cosine annealing formula
        # progress = 0 -> cosine_decay = 1.0 -> lr = base_lr
        # progress = 1 -> cosine_decay = 0.0 -> lr = final_lr
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.final_lr + (base_lr - self.final_lr) * cosine_decay

    def state_dict(self) -> Dict[str, Any]:
        """
        Return scheduler state for checkpointing.
        
        Returns:
            Dictionary containing scheduler state
        """
        return {
            'warmup_epochs': self.warmup_epochs,
            'total_epochs': self.total_epochs,
            'warmup_start_lr': self.warmup_start_lr,
            'final_lr': self.final_lr,
            'last_epoch': self.last_epoch,
            'base_lrs': self.base_lrs,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        """
        Load scheduler state from checkpoint.
        
        Args:
            state: Dictionary containing scheduler state
        """
        self.warmup_epochs = state['warmup_epochs']
        self.total_epochs = state['total_epochs']
        self.warmup_start_lr = state['warmup_start_lr']
        self.final_lr = state['final_lr']
        self.last_epoch = state['last_epoch']
        self.base_lrs = state['base_lrs']
        
        # Restore learning rates to match current epoch
        self._set_lr(self.last_epoch)
    
    def __repr__(self) -> str:
        """String representation of the scheduler."""
        return (f"{self.__class__.__name__}("
                f"warmup_epochs={self.warmup_epochs}, "
                f"total_epochs={self.total_epochs}, "
                f"warmup_start_lr={self.warmup_start_lr}, "
                f"final_lr={self.final_lr}, "
                f"last_epoch={self.last_epoch})")

class ModelEMA:
    """Exponential Moving Average (EMA) of model weights.

    Keep a copy of the model that is updated as an exponential moving average
    of the training model. Use the EMA weights for validation/inference to
    improve stability and performance.
    """

    def __init__(self, model: nn.Module, decay: float = os.environ.get("EMA_DECAY", 0.999)):
        self.ema = deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.collected_params = None
        self.initialized = False

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update EMA weights with current model weights."""
        if not self.initialized:
            self.ema.load_state_dict(model.state_dict())
            self.initialized = True
            return
        
        # ✅ FIX: 直接更新，不需要改变model的训练状态
        d = self.decay
        msd = model.state_dict()
        for k, v_ema in self.ema.state_dict().items():
            v_model = msd[k].detach()
            v_ema.copy_(v_ema * d + v_model * (1.0 - d))

    def state_dict(self) -> Dict[str, Any]:
        return {"decay": self.decay, "ema_state": self.ema.state_dict(), "initialized": self.initialized}

    def load_state_dict(self, state: Dict[str, Any]):
        self.decay = state.get("decay", self.decay)
        self.ema.load_state_dict(state["ema_state"])
        self.initialized = state.get("initialized", True)

    def store(self, model: nn.Module):
        """Save current model parameters for later restoration."""
        self.collected_params = [p.clone() for p in model.parameters()]

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        """Copy EMA parameters to the given model (for eval/export)."""
        for p_model, p_ema in zip(model.parameters(), self.ema.parameters()):
            p_model.copy_(p_ema)

    @torch.no_grad()
    def restore(self, model: nn.Module):
        """Restore model parameters saved by store()."""
        if self.collected_params is not None:
            for p, cp in zip(model.parameters(), self.collected_params):
                p.copy_(cp)
            self.collected_params = None

def cifar_stats() -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Return CIFAR-100 mean/std for normalization."""
    return (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)

def load_transforms() -> transforms.Compose:
    """Load the data transformations."""
    mean, std = cifar_stats()
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

def load_online_augmentation_transforms() -> transforms.Compose:
    """Load the online augmentation transforms."""
    mean, std = cifar_stats()
    erasing_prob = float(os.environ.get("RANDOM_ERASING_PROB", 0.5))
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=erasing_prob, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0.0, inplace=False),
    ])

def load_combined_data(augmented_train_dir: str, raw_train_dir: str, val_dir: str, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """
    Load the combined training and validation data.

    Args:
        augmented_train_dir: Directory containing offline augmented training data
        raw_train_dir: Directory containing raw training data (for online augmentation)
        val_dir: Directory containing validation data
        batch_size: Batch size for data loaders
    Returns:
        train_loader, val_loader: Combined training data loader and validation data loader
    """
    if os.environ.get("USE_ONLINE_AUGMENTATION", "1") == "1":
        online_augmented_dataset = datasets.ImageFolder(root=raw_train_dir, transform=load_online_augmentation_transforms())
    else:
        online_augmented_dataset = None
        
    offline_augmented_dataset = datasets.ImageFolder(root=augmented_train_dir, transform=load_transforms())

    if online_augmented_dataset is not None:
        combined_train_dataset = ConcatDataset([offline_augmented_dataset, online_augmented_dataset])
    else:
        combined_train_dataset = offline_augmented_dataset
    
    val_dataset = datasets.ImageFolder(root=val_dir, transform=load_transforms())

    pin = cuda.is_available()
    # Use min to avoid over-subscription on machines with few cores
    # Allow override via environment variable for flexibility
    num_workers = int(os.environ.get("NUM_WORKERS", min(os.cpu_count() or 1, 4)))
    
    train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)

    print("="*60)
    print(f"📊 Dataset Summary:")
    print(f"   Offline augmented data: {len(offline_augmented_dataset)} samples from {augmented_train_dir}")
    if online_augmented_dataset is not None:
        print(f"   Online augmented data: {len(online_augmented_dataset)} samples from {raw_train_dir}")
        print(f"   Offline and online augmented training dataset: {len(combined_train_dataset)} samples")
    else:
        print(f"   Online augmented data: Disabled")
        print(f"   Offline augmented training dataset: {len(combined_train_dataset)} samples from {augmented_train_dir}")
    if os.environ.get("MIXUP_ALPHA", "0.4") != "0" or os.environ.get("CUTMIX_ALPHA", "1.0") != "0":
        print(f"   MixUp/CutMix During Data Loading: Enabled")
        print(f"   MixUp alpha: {os.environ.get('MIXUP_ALPHA', '0.4')}")
        print(f"   CutMix alpha: {os.environ.get('CUTMIX_ALPHA', '1.0')}")
        print(f"   Mix probability: {os.environ.get('MIX_PROB', '0.5')}")
    else:
        print(f"   MixUp/CutMix: Disabled")
    print(f"   Validation dataset: {len(val_dataset)} samples from {val_dir}")
    print(f"   Number of classes: {len(val_dataset.classes)}")
    print(f"   Data loader workers: {num_workers}")
    print("="*60)
    return train_loader, val_loader

def define_loss_and_optimizer(model: nn.Module, lr: float, weight_decay: float) -> Tuple[nn.Module, optim.Optimizer, WarmupCosineScheduler]:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'bn' in name or 'bias' in name:
                no_decay.append(param)
            else:
                decay.append(param)     
    param_groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    
    ls = float(os.environ.get("LABEL_SMOOTHING", 0.1))
    criterion = LabelSmoothingCrossEntropy(smoothing=ls) if ls > 0 else nn.CrossEntropyLoss()
    optimizer = optim.SGD(param_groups, lr=lr, momentum=0.9, nesterov=True)
    scheduler = WarmupCosineScheduler(optimizer)
    
    print(f"   Loss: {criterion.__class__.__name__}")
    print(f"   Optimizer: {optimizer.__class__.__name__}")
    print(f"   Scheduler: {scheduler.__class__.__name__}")
    print(f"   Initial LR: {lr:.4f}")
    
    if hasattr(torch, 'compile') and torch.__version__ >= "2.0":
        try:
            model = torch.compile(model)
            print("🚀 Model compiled for better performance")
        except Exception as e:
            print(f"⚠️ Model compilation failed: {e}")
    
    return criterion, optimizer, scheduler

def _rand_bbox(size, lam):
    """Generate random bounding box for CutMix
    Args:
        size: Input tensor size (N, C, H, W)
        lam: Lambda value for mixing ratio
    Returns:
        bbx1, bby1, bbx2, bby2: Bounding box coordinates
    """
    H = size[2]
    W = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # randint(0, W) can return W, which is out of bounds for indexing
    # Use randint(0, W) to get values in [0, W-1]
    cx = np.random.randint(W) if W > 0 else 0
    cy = np.random.randint(H) if H > 0 else 0

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                optimizer: optim.Optimizer, device: torch.device, 
                scaler: Optional[GradScaler] = None, ema: Optional[ModelEMA] = None) -> Tuple[float, float]:
    """
    Train the model for one epoch with optimized mixed precision
    Args:
        model: The model to train
        dataloader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        scaler: Scaler for mixed precision training (optional, will be created if None)
        ema: Exponential Moving Average model (optional)
    Returns:
        Average loss and accuracy for the epoch (loss, accuracy)
    Note:
        - Scaler should be passed from the main training loop to maintain state (mutable object)
        - Accuracy with MixUp/CutMix uses original labels and may be lower than validation
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    amp_enabled = (device == 'cuda' or (hasattr(device, 'type') and device.type == 'cuda')) and os.environ.get("USE_AMP", "1") == "1"
    dev_type = 'cuda' if (device == 'cuda' or (hasattr(device, 'type') and device.type == 'cuda')) else 'cpu'
    
    # Scaler should be passed from main loop to maintain state across epochs
    # If not provided, create a new one (but this will lose history)
    if scaler is None:
        scaler = GradScaler(enabled=amp_enabled)
    
    max_norm = float(os.environ.get("CLIP_NORM", 1.0))
    
    mixup_alpha = float(os.environ.get("MIXUP_ALPHA", 0.4))
    cutmix_alpha = float(os.environ.get("CUTMIX_ALPHA", 1.0))
    mix_prob = float(os.environ.get("MIX_PROB", 0.5))

    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        # Save original labels for accuracy calculation
        original_labels = labels.clone()

        optimizer.zero_grad(set_to_none=True)

        # Apply MixUp or CutMix augmentation
        use_mixup = False
        use_cutmix = False
        lam = 1.0
        
        if model.training and (mixup_alpha > 0 or cutmix_alpha > 0):
            r = random.random()
            
            if mixup_alpha > 0 and cutmix_alpha > 0:
                if r < mix_prob:
                    use_mixup = True
                else:
                    use_cutmix = True
            elif mixup_alpha > 0:
                use_mixup = True
            elif cutmix_alpha > 0:
                use_cutmix = True
            
            if use_mixup:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                index = torch.randperm(inputs.size(0), device=device)
                inputs = lam * inputs + (1 - lam) * inputs[index, :]
                labels_a, labels_b = labels, labels[index]
            elif use_cutmix:
                lam = np.random.beta(cutmix_alpha, cutmix_alpha)
                index = torch.randperm(inputs.size(0), device=device)
                bbx1, bby1, bbx2, bby2 = _rand_bbox(inputs.size(), lam)
                inputs[:, :, bby1:bby2, bbx1:bbx2] = inputs[index, :, bby1:bby2, bbx1:bbx2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size(-1) * inputs.size(-2)))
                labels_a, labels_b = labels, labels[index]

        # Forward pass and loss calculation with autocast
        # PyTorch autocast automatically uses FP32 for loss functions to ensure numerical stability
        with autocast(device_type=dev_type, enabled=amp_enabled):
            outputs = model(inputs)
            
            # Loss calculation INSIDE autocast - PyTorch handles precision automatically
            if use_mixup or use_cutmix:
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            else:
                loss = criterion(outputs, labels)
        
        # Check for NaN or infinite loss values
        if not torch.isfinite(loss):
            print(f"Warning: Non-finite loss detected: {loss.item()}, skipping this batch")
            continue

        # Backward pass
        scaler.scale(loss).backward()
        
        if max_norm > 0.0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        # Check for gradient scaling issues
        if amp_enabled and scaler.get_scale() < 1e-8:
            print(f"Warning: Gradient scaler scale is very small: {scaler.get_scale()}")
            scaler.update()
            continue
        
        scaler.step(optimizer)
        scaler.update()

        # Update EMA after optimizer step
        if ema is not None:
            ema.update(model)

        # Statistics - use original labels for accuracy
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += original_labels.size(0)
        # Use original labels for accuracy calculation
        correct += predicted.eq(original_labels).sum().item()

        progress_bar.set_postfix(
            {"Loss": f"{loss.item():.4f}", "Acc": f"{100.0 * correct / total:.2f}%", "LR": f"{optimizer.param_groups[0]['lr']:.6f}"}
        )

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def validate_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    amp_enabled = (device == 'cuda' or (hasattr(device, 'type') and device.type == 'cuda')) and os.environ.get("USE_AMP", "1") == "1"
    dev_type = 'cuda' if (device == 'cuda' or (hasattr(device, 'type') and device.type == 'cuda')) else 'cpu'

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation", leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            with autocast(device_type=dev_type, enabled=amp_enabled):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix(
                {"Loss": f"{loss.item():.4f}", "Acc": f"{100.0 * correct / total:.2f}%"}
            )

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def save_checkpoint(state: Dict[str, Any], filename: str):
    """
    Save model checkpoint
    Args:
        state: Checkpoint state
        filename: Path to save checkpoint
    """
    torch.save(state, filename)


def load_checkpoint(filename: str, model: nn.Module, optimizer: Optional[optim.Optimizer] = None, 
                    scheduler: Optional[WarmupCosineScheduler] = None, device: Optional[str] = None) -> Dict[str, Any]:
    """
    Load model checkpoint
    Args:
        filename: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to map checkpoint to (optional, defaults to CPU)
    Returns:
        Checkpoint state
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Checkpoint file {filename} not found")

    # Add map_location for cross-device compatibility
    if device is None:
        device = 'cpu'
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint

def save_metrics(metrics: str, filename: str = "training_metrics.txt"):
    """
    Save training metrics to a file
    Args:
        metrics: Metrics string to save
        filename: Path to save metrics
    """
    with open(filename, 'w') as f:
        f.write(metrics)