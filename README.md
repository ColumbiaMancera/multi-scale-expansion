# Multi Scale Expansion
Repository to create framework for image classification computer vision models in tensorflow. <br>
[![PyPI](https://img.shields.io/pypi/v/multi-scale-expansion)](https://pypi.org/project/multi-scale-expansion/)
![image](https://img.shields.io/pypi/l/tensorflow)
![image](https://img.shields.io/github/issues/ColumbiaMancera/multi-scale-expansion)
![Build Status](https://github.com/ColumbiaMancera/multi-scale-expansion/actions/workflows/build.yml/badge.svg)
[![codecov](https://codecov.io/gh/ColumbiaMancera/multi-scale-expansion/branch/main/graph/badge.svg)](https://codecov.io/gh/ColumbiaMancera/multi-scale-expansion)
[![Docs](https://img.shields.io/badge/docs-passing-success)](https://columbiamancera.github.io/multi-scale-expansion/)

## Overview
`multi-scale-expansion` is a library for automating the set up of an image classification model. The user provides their data, and the library creates and trains a ready-to-use model to complete the image classification task and apply it to any image further. The objective is that this framework can be automated and applied for recognizing whether a plant is healthy or not, through the use of the models we train. 

## Contributions
For instructions on how to contribute, go to the [Contribution Guidelines Page](https://github.com/ColumbiaMancera/multi-scale-expansion/blob/main/CONTRIBUTING.md). 

## Installation
Prerequisites: 
- Python >= 3.7
- Torch & Torchvision
- Numpy 
- Matplotlib
- PIL (Pillow) 

To install Python packages: 
```bash
$ pip install torch
$ pip install torchvision
$ pip install numpy
$ pip install matplotlib
$ pip install Pillow
```

To install library: 
```bash
$ pip install multi-scale-expansion
```

## Quick-Start Example 
```python 

# Provide a pytorch pre-trained image classification model! 
mock_model = ms_model.get_plant_model(mock_model, list(range(6))

# Specify these values - fine-tune at your will!
mock_criterion, mock_optimizer, mock_lr_scheduler = get_train_loss_needs(
    mock_model, mock_lr, mock_momentum, mock_step_size, mock_gamma
)

# Get dataloaders from datasets
mock_dataloaders = ms_datasets.get_dataloaders(mock_datasets)

# Fine-tune your pre-trained model! 
model, train_losses, train_accuracies, val_losses, val_accuracies = ms.train_model(
    device,
    mock_dataset_sizes,
    mock_dataloaders,
    mock_model,
    mock_criterion,
    mock_optimizer,
    mock_lr_scheduler,
    num_epochs=1,
    testing=True,
)
```

And now your model is ready-to-use for your image classification task! 
Soon, you'll be able to just call a single method and the library will set up the whole classification task for you! 
