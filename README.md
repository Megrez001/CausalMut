# CausalMut
CausalMut: A Causal Deep Learning Framework for Enzyme Mutational Effect Prediction

## Overview

CausalMut is a deep learning framework that integrates multi-modal features (protein sequences, molecular structures, and mutation information) with a causal-inspired architecture to predict enzyme mutant catalytic efficiency .
![Framework](figure/fig3.png)

## Pre-trained Models Preparation

### ESM-2 Model
```bash
wget https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt

### MolT5 Model
```bash
git clone https://huggingface.co/laituan245/molt5-base
```
