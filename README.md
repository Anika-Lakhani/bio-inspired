# Bio-Inspired CNN: Robust Neural Networks Through Biological Mechanisms (README Generated with Claude)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.7%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-1.9%2B-orange)

A PyTorch implementation of biologically-inspired convolutional neural networks that demonstrate improved robustness to noise and occlusion through the incorporation of lateral inhibition and attention mechanisms.

## Overview

This repository contains a PyTorch implementation of biologically-inspired CNNs that incorporate lateral inhibition and attention mechanisms observed in the mammalian visual cortex. These mechanisms are implemented as modular components that can be added to standard CNN architectures to improve their robustness against various types of image degradation, such as noise and occlusion.

The biological inspiration comes from two key mechanisms:

1. **Lateral Inhibition**: Implements center-surround receptive fields with a Difference of Gaussians (DoG) kernel, mimicking the inhibitory connections found in the visual cortex that enhance contrast and edge detection.

2. **Biological Attention**: Combines channel and spatial attention mechanisms to selectively enhance important features, inspired by attentional processes in the visual system.

## Key Features

- Modular implementation of biological mechanisms that can be incorporated into any CNN architecture
- Comparative analysis framework for testing different model configurations
- Robustness tests against Gaussian noise and occlusion
- Ablation study components for evaluating the individual contribution of each biological mechanism
- Comprehensive visualization tools for model performance analysis

## Code Demo

To familiarize yourself with the codebase, first watch the optional code demo at tinyurl.com/anika-bio-inspired (also linked in START-HERE.png in repo)

## Installation

```bash
git clone https://github.com/Anika-Lakhani/bio-inspired.git
cd bio-inspired
pip install -r requirements.txt
```

## Usage

### Basic Training and Evaluation

```python
from models import BioInspiredCNN, ControlCNN
from experiments import run_experiment

# Run the experiment with ablation study
results = run_experiment()

# Results will be saved in the ./results directory
```

### Custom Model Configuration

```python
from models import BioInspiredCNN

# Create a model with only lateral inhibition
model_inhibition = BioInspiredCNN(
    placement='middle',
    use_inhibition=True,
    use_attention=False
)

# Create a model with only biological attention
model_attention = BioInspiredCNN(
    placement='middle', 
    use_inhibition=False, 
    use_attention=True
)

# Create a combined model
model_combined = BioInspiredCNN(
    placement='middle',
    use_inhibition=True,
    use_attention=True
)
```

## Model Architecture

The bio-inspired models extend a base CNN architecture (MobileNetV3-Small in the current implementation) with specialized biological mechanisms:

1. **Lateral Inhibition Layer**: Implements Difference of Gaussians (DoG) kernel for enhanced edge detection and noise suppression
2. **Biological Attention Module**: Combines channel and spatial attention to focus on relevant features
3. **Bio-Inspired Block**: Integrates both mechanisms with adaptive mixing parameters

## Experimental Results

The repository includes comprehensive experiments comparing different model configurations:

- **Control CNN**: Standard MobileNetV3-Small without biological mechanisms
- **Bio_Combined**: Complete implementation with both lateral inhibition and attention
- **Bio_Inhibition_Only**: Implementation with only lateral inhibition
- **Bio_Attention_Only**: Implementation with only attention mechanisms

Experiments evaluate model performance on:
- Clean images
- Images with Gaussian noise
- Images with occlusion

Results demonstrate that biologically-inspired mechanisms improve robustness to noise and occlusion, with different mechanisms providing different benefits.

## Citation

If you use this code in your research, please cite:

```
@misc{lakhani2025bioinspired,
  author = {Lakhani, Anika},
  title = {Bio-Inspired CNN: Robust Neural Networks Through Biological Mechanisms},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Anika-Lakhani/bio-inspired}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Implementations inspired by research in computational neuroscience and bio-inspired computing
- Base CNN architectures from torchvision library
