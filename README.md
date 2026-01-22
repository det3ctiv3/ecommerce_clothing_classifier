# Fashion MNIST Clothing Classifier

A convolutional neural network (CNN) for classifying clothing items from the Fashion MNIST dataset. 

## Overview

This project implements a simple CNN to classify grayscale images of clothing into 10 categories:  T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot. 

## Requirements

```
torch
torchvision
torchmetrics
numpy
matplotlib
```

Install dependencies:
```bash
pip install torch torchvision torchmetrics numpy matplotlib
```

## Dataset

Fashion MNIST: 
- 60,000 training images
- 10,000 test images
- Image size: 28×28 grayscale
- 10 clothing categories

The dataset is automatically downloaded on first run.

## Model Architecture

```
Input (1×28×28)
    ↓
Conv2d (1→16 channels, 3×3 kernel, padding=1)
    ↓
ReLU
    ↓
MaxPool2d (2×2)
    ↓
Flatten
    ↓
Linear (3136→10)
    ↓
Output (10 classes)
```

## Usage

### Training

```python
python train. py
```

Default hyperparameters:
- Batch size: 10
- Learning rate: 0.001
- Optimizer: Adam
- Loss function: CrossEntropyLoss
- Epochs: 1 (configurable)

### Evaluation

The script automatically evaluates the model on the test set after training and reports:
- Overall accuracy
- Per-class precision
- Per-class recall

## Configuration

Key parameters defined at the top of the script:

```python
num_input_channels = 1    # Grayscale images
num_output_channels = 16  # Conv layer output channels
image_size = 28           # Input image dimensions
num_classes = 10          # Number of clothing categories
```

## Code Structure

```
├── Data loading (FashionMNIST)
├── Model definition (ClothesClassifier)
├── Training function
├── Model initialization
├── Training loop
└── Evaluation metrics
```

## Output

Training output:
```
epoch 0, loss: 0. XXXX
```

Evaluation metrics:
```
Accuracy: 0.XX
Precision (per class): [0.XX, 0.XX, ...]
Recall (per class): [0.XX, 0.XX, ...]
```

## Notes

- The model expects input shape `[batch, 1, 28, 28]`
- Training on CPU is feasible due to small image size
- Consider increasing epochs for better performance
- Model checkpointing is not implemented

## License

MIT
