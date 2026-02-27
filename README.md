# Disentangled-Interpretable-Self-supervised-Representation-Learning-for-Medical-Images

This repository contains the official implementation of our MICCAI 2026 submission.

## Overview

We propose an **augmentation-disentangled SSL pre-training framework** that assigns each augmentation type a dedicated projection head, trained through a factored contrastive loss. A residual head preserves shared semantic information across all transformations. To enable task-adaptive feature selection, we introduce **Differentiable Augmentation Search (DAUS)**, which searches over discrete subspace combinations to identify the optimal augmentation subset for each downstream task.

## Installation

```bash
conda env create -f environment.yaml
conda activate DCRL
cd bin
```

## Quick run guide

### Pre-training

**Baseline (supervised):**
```bash
python main_train.py \
  --mode baseline \
  --dataset derma \
  --backbone resnet34 \
  --pretrained \
  --medmnist_size 224
```

**Standard contrastive (e.g., SimCLR):**
```bash
python main_train.py \
  --mode contrastive \
  --dataset derma \
  --backbone resnet34 \
  --pretrained \
  --contrastive_loss simclr \
  --medmnist_size 224
```

**Ours (disentangled):**
```bash
python main_train.py \
  --mode disentangled \
  --dataset derma \
  --backbone resnet34 \
  --pretrained \
  --contrastive_loss simclr \
  --num_aug_groups 6 \
  --group_size 128 \
  --inactive_weight 1.0 \
  --ortho_weight 0.5 \
  --uniform_weight 0.1 \
  --selection_method darts \
  --discretization_method topk \
  --warmup_epochs 10 \
  --medmnist_size 224
```

### Class-Incremental Learning

```bash
python main_train.py \
  --mode disentangled \
  --dataset organ \
  --backbone resnet34 \
  --pretrained \
  --class_incremental \
  --medmnist_size 224
```

---

---

> **Note:** We are currently cleaning up the codebase for public release. The full code will be updated after cleaning.
