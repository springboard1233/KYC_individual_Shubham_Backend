#!/bin/bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CPU only)
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2

# Install PyG dependencies (match PyTorch 2.1.2)
pip install torch-geometric==2.6.1

# Install the rest of your dependencies
pip install -r requirements.txt
