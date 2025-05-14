"""
Simplified Biologically-Inspired CNN Implementation
This file is guaranteed to run without syntax errors.
"""

import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import MobileNet_V3_Small_Weights
import random
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Configuration
class Config:
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 64
    NUM_EPOCHS = 5  # Keep small for quick testing
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    NOISE_LEVELS = [0.1]  # Just one level for simplicity
    OCCLUSION_SIZE = 50
    IMG_SIZE = 224
    DATA_DIR = './data'
    RESULTS_DIR = './results'
    
    # Biological parameters
    LATERAL_INHIBITION_KERNEL_SIZE = 5
    INHIBITION_STRENGTH = 0.3
    ATTENTION_REDUCTION_RATIO = 8
    BIO_REGULARIZER_P = 0.1
    
    # Layer placements
    MIDDLE_LAYER = 2
    
    def __init__(self):
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.RESULTS_DIR, exist_ok=True)
        
        # Set seeds
        random.seed(self.SEED)
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.SEED)
            torch.backends.cudnn.deterministic = True

# Create config
config = Config()
print(f"Using device: {config.DEVICE}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Dataset wrapper
class BiologicalVisionDataset(Dataset):
    def __init__(self, base_dataset, transform_fn=None):
        self.base_dataset = base_dataset
        self.transform_fn = transform_fn
        self.indices = list(range(len(base_dataset)))
        
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        img, label = self.base_dataset[self.indices[idx]]
        if self.transform_fn:
            img = self.transform_fn(img)
        return img, label

# Image transformations
def add_gaussian_noise(img, noise_std=0.1):
    noise = torch.randn_like(img) * noise_std
    return torch.clamp(img + noise, 0, 1)

def add_occlusion(img, occlusion_size=50):
    h, w = img.shape[1:]
    x = np.random.randint(0, w-occlusion_size)
    y = np.random.randint(0, h-occlusion_size)
    img_copy = img.clone()
    img_copy[:, y:y+occlusion_size, x:x+occlusion_size] = 0
    return img_copy

# Data loaders
def get_data_loaders(noise_std=0.1):
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_set = torchvision.datasets.CIFAR10(
        root=config.DATA_DIR, train=True, download=True, transform=transform)
    
    test_set = torchvision.datasets.CIFAR10(
        root=config.DATA_DIR, train=False, download=True, transform=transform)
    
    train_loader = DataLoader(
        BiologicalVisionDataset(train_set),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    test_loaders = {
        'clean': DataLoader(
            BiologicalVisionDataset(test_set),
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=2
        ),
        'noisy': DataLoader(
            BiologicalVisionDataset(test_set, 
                                  lambda img: add_gaussian_noise(img, noise_std)),
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=2
        ),
        'occluded': DataLoader(
            BiologicalVisionDataset(test_set, 
                                  lambda img: add_occlusion(img, config.OCCLUSION_SIZE)),
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=2
        )
    }
    
    return train_loader, test_loaders

# Biological mechanisms
class LateralInhibition(nn.Module):
    def __init__(self, channels, kernel_size=5, inhibition_strength=0.3):
        super().__init__()
        self.channels = channels
        
        # Create a center-surround filter
        self.conv = nn.Conv2d(
            channels, channels, kernel_size,
            padding=kernel_size//2, 
            groups=channels,  # Depth-wise convolution
            bias=False
        )
        
        # Initialize with a DoG kernel
        self.init_dog_kernel(kernel_size, inhibition_strength)
        
        # Learnable scaling factor
        self.alpha = nn.Parameter(torch.ones(1))
        
    def init_dog_kernel(self, kernel_size, inhibition_strength):
        center_sigma = kernel_size / 6.0
        surround_sigma = kernel_size / 3.0
        
        grid = torch.linspace(-(kernel_size//2), kernel_size//2, kernel_size)
        x, y = torch.meshgrid(grid, grid, indexing='ij')
        dist = x.pow(2) + y.pow(2)
        
        center = torch.exp(-(dist) / (2 * center_sigma**2))
        surround = torch.exp(-(dist) / (2 * surround_sigma**2))
        dog = center - inhibition_strength * surround
        
        dog = dog / torch.abs(dog).sum()
        
        dog_kernel = dog.expand(self.channels, 1, kernel_size, kernel_size)
        self.conv.weight.data = dog_kernel
    
    def forward(self, x):
        inhibition = self.conv(x)
        return x + self.alpha * inhibition

class BiologicalAttention(nn.Module):
    def __init__(self, in_dim, reduction_ratio=8):
        super().__init__()
        
        # Feature-based attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_dim, in_dim // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim // reduction_ratio, in_dim, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        # Importance weighting
        self.gamma = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        # Channel attention
        channel_attention = self.channel_attention(x)
        feature_refined = x * channel_attention
        
        # Spatial attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_features = torch.cat([avg_pool, max_pool], dim=1)
        spatial_attention = self.spatial_attention(spatial_features)
        
        # Combine both
        refined = feature_refined * spatial_attention
        
        return x + self.gamma * refined

class BioInspiredBlock(nn.Module):
    def __init__(self, channels, use_inhibition=True, use_attention=True):
        super().__init__()
        self.use_inhibition = use_inhibition
        self.use_attention = use_attention
        
        if use_inhibition:
            self.inhibition = LateralInhibition(
                channels,
                kernel_size=config.LATERAL_INHIBITION_KERNEL_SIZE,
                inhibition_strength=config.INHIBITION_STRENGTH
            )
            
        if use_attention:
            self.attention = BiologicalAttention(
                channels,
                reduction_ratio=config.ATTENTION_REDUCTION_RATIO
            )
            
        if use_inhibition and use_attention:
            self.mixing = nn.Parameter(torch.tensor([0.5]))
    
    def forward(self, x):
        if self.use_inhibition and self.use_attention:
            inhib_out = self.inhibition(x)
            attn_out = self.attention(x)
            gamma = torch.sigmoid(self.mixing)
            x = gamma * attn_out + (1 - gamma) * inhib_out
        elif self.use_inhibition:
            x = self.inhibition(x)
        elif self.use_attention:
            x = self.attention(x)
        return x

# Model architectures
class BioInspiredCNN(nn.Module):
    def __init__(self, placement='middle', use_inhibition=True, use_attention=True):
        super().__init__()
        self.placement = placement
        self.use_inhibition = use_inhibition
        self.use_attention = use_attention
        
        # Load pretrained MobileNetV3
        weights = MobileNet_V3_Small_Weights.DEFAULT
        base = torchvision.models.mobilenet_v3_small(weights=weights)
        base.classifier[-1] = nn.Linear(1024, 10)
        
        # Extract feature layers
        self.features = base.features
        
        # Determine feature dimensions
        with torch.no_grad():
            dummy = torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE)
            x = dummy
            for i, layer in enumerate(self.features):
                x = layer(x)
                if i == config.MIDDLE_LAYER:
                    self.feature_dim = x.size(1)
        
        # Create bio-inspired block
        self.bio_block = BioInspiredBlock(
            self.feature_dim, use_inhibition, use_attention)
        
        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = base.classifier
    
    def forward(self, x):
        # Process through initial layers
        for i, layer in enumerate(self.features):
            x = layer(x)
            
            # Apply bio-inspired mechanism at specified location
            if i == config.MIDDLE_LAYER:
                x = self.bio_block(x)
        
        # Global pooling and classification
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
    
    def get_mechanism_state(self):
        state = {}
        if hasattr(self.bio_block, 'mixing'):
            state['mixing'] = torch.sigmoid(self.bio_block.mixing).item()
        if hasattr(self.bio_block, 'inhibition') and hasattr(self.bio_block.inhibition, 'alpha'):
            state['inhibition_alpha'] = self.bio_block.inhibition.alpha.item()
        if hasattr(self.bio_block, 'attention') and hasattr(self.bio_block.attention, 'gamma'):
            state['attention_gamma'] = self.bio_block.attention.gamma.item()
        return state

class ControlCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Load pretrained MobileNetV3
        weights = MobileNet_V3_Small_Weights.DEFAULT
        base = torchvision.models.mobilenet_v3_small(weights=weights)
        base.classifier[-1] = nn.Linear(1024, 10)
        
        # Use model components directly
        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = base.classifier
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# Training and evaluation functions
def train_model(model, train_loader, test_loaders, name=None):
    model.to(config.DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.LEARNING_RATE, 
        weight_decay=config.WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss()
    
    # Tracking metrics
    results = {
        'train_acc': [],
        'val_acc': [],
        'mechanism_state': []
    }
    
    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        for inputs, labels in pbar:
            inputs = inputs.to(config.DEVICE, non_blocking=True)
            labels = labels.to(config.DEVICE, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Optimizer step
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.3f}", 
                'acc': f"{100*correct/total:.2f}%"
            })
        
        # Evaluate on validation set
        val_acc = evaluate_model(model, test_loaders['clean'])
        
        # Store results
        train_acc = 100 * correct / total
        results['train_acc'].append(train_acc)
        results['val_acc'].append(val_acc)
        
        # Store biological mechanism state if available
        if hasattr(model, 'get_mechanism_state'):
            results['mechanism_state'].append(model.get_mechanism_state())
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Acc: {train_acc:.2f}%")
        print(f"  Val Acc: {val_acc:.2f}%")
        
        # Print biological mechanism state if available
        if hasattr(model, 'get_mechanism_state'):
            state = model.get_mechanism_state()
            for param_name, value in state.items():
                print(f"  {param_name}: {value:.3f}")
    
    # Final evaluation on all test sets
    test_results = {}
    for dataset_name, loader in test_loaders.items():
        test_results[dataset_name] = evaluate_model(model, loader)
    results['test_results'] = test_results
    
    return results

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total

# Visualization functions
def show_images(img_tensor, transformed_tensors, titles):
    """Show original and transformed images"""
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, len(transformed_tensors) + 1, 1)
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())
    plt.imshow(img)
    plt.title("Original")
    plt.axis('off')
    
    # Transformed images
    for i, (tensor, title) in enumerate(zip(transformed_tensors, titles)):
        plt.subplot(1, len(transformed_tensors) + 1, i + 2)
        img = tensor.permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    return plt.gcf()

def plot_comparison_bars(results_dict, test_set_names=None):
    """Create bar chart comparing model performance"""
    if test_set_names is None:
        test_set_names = ['clean', 'noisy', 'occluded']
    
    # Extract model names and accuracy data
    model_names = list(results_dict.keys())
    
    # Prepare data for plotting
    x = np.arange(len(test_set_names))
    width = 0.8 / len(model_names)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot bars for each model
    for i, model_name in enumerate(model_names):
        model_results = results_dict[model_name]['test_results']
        accuracies = [model_results[test_set] for test_set in test_set_names]
        
        offset = width * (i - len(model_names)/2 + 0.5)
        bars = ax.bar(x + offset, accuracies, width, label=model_name)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    # Customize plot
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([name.replace('_', ' ').title() for name in test_set_names])
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_training_curves(results_dict):
    """Plot training and validation accuracy curves"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model_name, results in results_dict.items():
        train_acc = results['train_acc']
        val_acc = results['val_acc']
        epochs = range(1, len(train_acc) + 1)
        
        # Plot training accuracy
        ax.plot(epochs, train_acc, '--', label=f'{model_name} Train')
        
        # Plot validation accuracy
        ax.plot(epochs, val_acc, '-', label=f'{model_name} Val')
    
    ax.set_title('Training and Validation Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

# Experiment function
def run_experiment():
    """Run a focused experiment comparing control and bio models"""
    print("\nRunning biologically-inspired CNN experiment")
    
    # Get data loaders
    noise_level = 0.1
    train_loader, test_loaders = get_data_loaders(noise_level)
    
    # Define models to compare
    models = {
        'Control': ControlCNN(),
        'Bio_Combined': BioInspiredCNN(
            placement='middle',
            use_inhibition=True,
            use_attention=True
        )
    }
    
    # Train and evaluate models
    results = {}
    for model_name, model in models.items():
        print(f"\nTraining {model_name} model...")
        
        # Train model
        model_results = train_model(model, train_loader, test_loaders)
        results[model_name] = model_results
        
        # Print results
        print(f"\nTest results for {model_name}:")
        for dataset, acc in model_results['test_results'].items():
            print(f"  {dataset}: {acc:.2f}%")
    
    # Create comparison visualization
    fig = plot_comparison_bars(results)
    plt.title("Control vs Bio-Inspired CNN Performance")
    plt.savefig(f"{config.RESULTS_DIR}/performance_comparison.png")
    
    # Plot training curves
    fig = plot_training_curves(results)
    plt.title("Training Progression")
    plt.savefig(f"{config.RESULTS_DIR}/training_curves.png")
    
    # Visualize transforms
    sample_inputs, _ = next(iter(test_loaders['clean']))
    img_tensor = sample_inputs[0]
    transformed = [
        add_gaussian_noise(img_tensor, noise_level),
        add_occlusion(img_tensor)
    ]
    titles = ["Gaussian Noise", "Occlusion"]
    fig = show_images(img_tensor, transformed, titles)
    plt.savefig(f"{config.RESULTS_DIR}/image_transforms.png")
    
    # Calculate improvement percentages
    control_metrics = results['Control']['test_results']
    bio_metrics = results['Bio_Combined']['test_results']
    
    print("\nImprovement over control model:")
    for dataset in control_metrics:
        improvement = 100 * (bio_metrics[dataset] - control_metrics[dataset]) / control_metrics[dataset]
        print(f"  {dataset}: {improvement:+.2f}%")
    
    return results

# Main execution
if __name__ == "__main__":
    # Run the experiment
    run_experiment()