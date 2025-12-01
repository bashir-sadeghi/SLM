# Simple RNN Language Model on WikiText-2

This repository contains a from-scratch implementation of a **recurrent neural network (RNN) language model** trained on the [WikiText-2](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/) dataset using **PyTorch**.

The goal of this project is to demonstrate how to build a small but fully working language model without relying on high-level training frameworks: manual data loading, batching, training loop, evaluation, and text generation.

---

## Features

- Token/character-level RNN language model (configurable embedding size, hidden size, number of layers, dropout)
- Training and validation loss tracking
- Checkpoint saving/loading
- Simple text generation from a trained model
- Clean, readable PyTorch code suitable as a reference or teaching example

---

## Requirements

- Python 3.9+
- PyTorch
- NumPy
- (Optional) tqdm / matplotlib for progress or plotting

Example `requirements.txt` (adjust to your exact versions if needed):

```txt
torch
numpy
tqdm
matplotlib
