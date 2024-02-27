# SimCLR for Audio Events

This repository contains the implementation of SimCLR (Simple Framework for Contrastive Learning of Visual Representations) adapted for audio event recognition. Our adaptation focuses on leveraging the power of contrastive learning to enhance feature extraction from audio data, enabling more accurate classification of audio events.

## Introduction

SimCLR is a simple yet powerful framework for learning visual representations by maximizing agreement between differently augmented views of the same data example via a contrastive loss. This project adapts SimCLR's methodology to the audio domain, addressing the unique challenges of audio data processing and representation.

## Features

- **Efficient Audio Processing Pipeline:** Customizable preprocessing for converting audio signals into suitable representations for contrastive learning.
- **Flexible Data Augmentation:** A set of audio-specific augmentations to generate varied views of the same audio event, enhancing the robustness of learned representations.
- **Scalable Contrastive Learning Framework:** Implementation supports running on multiple GPUs to handle large datasets and extensive training periods.
- **Pretrained Models:** Access to models pretrained on popular audio datasets, ready for fine-tuning on your specific audio recognition tasks.

## Installation

To set up the environment for running this project, follow these steps:

```bash
git clone --recursive https://github.com/kligvasser/simclr-audio-events.git
cd simclr-audio-events
conda env create -f environment.yaml
```

## Usage
- Prepare your dataset according to the expected format (detailed in the notebooks/built-dataframes.ipynb).
- Configure your training parameters in config.yaml.
- Run the training script:
```bash
accelerate launch --config_file configs/accelerate.yaml train.py --config ./configs/simclr-audioset-vitb32.yaml
```

## External links
https://drive.google.com/drive/folders/1l9Q7hEfWZDiS6P0NLrNvzSRT63f_wWH-?usp=sharing