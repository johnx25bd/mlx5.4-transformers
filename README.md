# MLX5 Weeks 4/5: MNIST Image Captioning with Transformers from Scratch

This project explores the implementation of transformer architectures from the ground up, culminating in using a Visual Transformer encoder and a Text Transformer decoder for captioning MNIST digits. Built as a learning exercise to understand the inner workings of transformer models, this project combines computer vision and natural language processing concepts.

## Overview

The project implements a transformer-based model that:
- Takes four joined MNIST digit images as input
- Processes them through a custom image encoder
- Generates sequential digit descriptions using a decoder with self-attention
- Demonstrates multi-modal learning by combining image and text features

## Technical Implementation

### Core Components
- **Image Encoder**: Processes 16x14x14 image patches into embeddings
- **Label Encoder**: Handles sequential digit labels with positional encoding
- **Multi-Head Attention**: Custom implementation of transformer attention mechanisms
- **Combined Model**: Integrates both encoders for end-to-end training

### Technologies Used
- **PyTorch**: Primary deep learning framework
- **Weights & Biases**: Experiment tracking and visualization
- **NumPy**: Numerical computations and array operations
- **MNIST Dataset**: Training data source

## Learning Outcomes

This project was built as an educational exercise to understand:
- Transformer architecture internals
- Multi-modal learning approaches
- Attention mechanisms
- PyTorch implementation patterns
- Deep learning training workflows

## Note

This is a learning project focused on understanding transformer architectures and is not intended for production use. The code may not be fully optimized or production-ready, but serves as a demonstration of implementing transformer concepts from scratch in PyTorch.

## Development Process

The project evolved through several iterations:
- Initial implementation of basic attention mechanisms
- Addition of residual connections and layer normalization
- Integration of learning rate scheduling
- Experimentation with different model architectures and hyperparameters
- Implementation of multi-head attention and positional encoding

## Tracking and Visualization

Training progress and model performance were tracked using Weights & Biases, allowing for:
- Loss and accuracy monitoring
- Learning rate adjustment visualization
- Model behavior analysis across training epochs