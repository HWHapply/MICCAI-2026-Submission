import warnings
warnings.filterwarnings("ignore", message="pkg_resources")
warnings.filterwarnings('ignore', message='xFormers is not available')
warnings.filterwarnings('ignore', message='Checkpoint directory.*exists and is not empty')
import torch
import os
torch.set_float32_matmul_precision('high')
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
torch.backends.cuda.matmul.fp32_precision = "tf32"
torch.backends.cudnn.conv.fp32_precision = "tf32"
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import argparse
import json
import logging
from pathlib import Path
from tqdm import tqdm
from itertools import combinations

import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, TQDMProgressBar

try:
    import monai.networks.nets as monai_nets
    MONAI_AVAILABLE = True
except:
    MONAI_AVAILABLE = False

import medmnist
from medmnist import INFO
from PIL import Image

from dataset import SimCLRAugmentation, ContrastiveDataset, create_disentangled_collate_fn, SliceAggregator
from SimCLR import SimCLR
from BarlowTwins import BarlowTwins
from VICReg import VICReg
from BYOL import BYOL
from MoCo import MoCo
from utils import generate_project_name, plot_roc_curve, plot_confusion_matrix, to_python


def get_medmnist_dataset(dataset_name: str, split: str = 'train', download: bool = True, size: int = None):
    """Load a MedMNIST dataset.

    Args:
        dataset_name: Name of the dataset
        split: 'train', 'val', or 'test'
        download: Whether to download if not present
        size: Image size for MedMNIST+ (28, 64, 128, 224). If None, uses default (28 for most datasets)

    Returns:
        dataset: MedMNIST dataset
        info: Dataset info dict (with added 'actual_size' field)
    """
    dataset_mapping = {
        'path': 'pathmnist', 'chest': 'chestmnist', 'pneumonia': 'pneumoniamnist',
        'derma': 'dermamnist', 'oct': 'octmnist', 'retina': 'retinamnist',
        'breast': 'breastmnist', 'blood': 'bloodmnist', 'tissue': 'tissuemnist',
        'organ': 'organamnist', 'organc': 'organcmnist', 'organs': 'organsmnist',
        'nodule': 'nodulemnist3d', 'adrenal': 'adrenalmnist3d', 'fracture': 'fracturemnist3d',
        'vessel': 'vesselmnist3d', 'synapse': 'synapsemnist3d', 'organmnist': 'organmnist3d',
    }

    dataset_name = dataset_name.lower()
    if dataset_name in dataset_mapping:
        dataset_name = dataset_mapping[dataset_name]

    info = INFO[dataset_name]
    DataClass = getattr(medmnist, info['python_class'])

    is_3d = '3d' in dataset_name.lower()

    # Load dataset with specified size (MedMNIST+ support)
    if size is not None:
        dataset = DataClass(split=split, download=download, as_rgb=True, size=size)
        info['actual_size'] = size
    else:
        dataset = DataClass(split=split, download=download, as_rgb=True)
        # Default size is 28 for most MedMNIST datasets
        info['actual_size'] = 28

    info['spatial_dims'] = 3 if is_3d else 2
    info['is_rgb'] = info['n_channels'] == 3

    return dataset, info


class BaselineModel(pl.LightningModule):
    """Baseline supervised learning model."""
    
    def __init__(self, backbone: str = 'resnet50', pretrained: bool = False,
                 freeze_backbone: bool = False, num_classes: int = 7,
                 learning_rate: float = 3e-4, weight_decay: float = 1e-4,
                 max_epochs: int = 500, spatial_dims: int = 2,
                 use_2d_for_3d: bool = False, slice_aggregation: str = 'mean',
                 hidden_dims: list = [512, 256], dropout: float = 0.5,
                 average: str = 'macro', projection_dim: int = 512, task: str = 'multi-class'):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.average = average
        self.freeze_backbone = freeze_backbone
        self.task = task
        
        self.encoder = self._get_backbone(backbone, pretrained, spatial_dims, use_2d_for_3d, slice_aggregation)
        
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print(f"✓ Backbone frozen")
        
        with torch.no_grad():
            if spatial_dims == 2:
                dummy_input = torch.zeros(1, 3, 224, 224)
            elif spatial_dims == 3:
                dummy_input = torch.zeros(1, 3, 28, 224, 224) if use_2d_for_3d else torch.zeros(1, 3, 28, 28, 28)
            encoder_dim = self.encoder(dummy_input).shape[1]
        
        if encoder_dim != projection_dim:
            self.projection = nn.Linear(encoder_dim, projection_dim)
        else:
            self.projection = nn.Identity()
        
        layers = []
        prev_dim = projection_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

        # Use appropriate loss function based on task type
        if task == 'multi-label':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.validation_step_outputs = []
        self.test_step_outputs = []

    def _get_backbone(self, backbone_name, pretrained, spatial_dims, use_2d_for_3d, slice_aggregation):
        if spatial_dims == 2:
            return self._get_2d_backbone(backbone_name, pretrained)
        elif spatial_dims == 3:
            if use_2d_for_3d:
                backbone_2d = self._get_2d_backbone(backbone_name, pretrained)
                return SliceAggregator(backbone_2d, aggregation_method=slice_aggregation)
            else:
                return self._get_3d_backbone(backbone_name, pretrained)
    
    def _get_2d_backbone(self, backbone_name, pretrained):
        if backbone_name.startswith('dinov2'):
            backbone = torch.hub.load('facebookresearch/dinov2', backbone_name, pretrained=pretrained)
            return backbone
        
        weights_map = {
            'resnet18': ResNet18_Weights.IMAGENET1K_V1 if pretrained else None,
            'resnet34': ResNet34_Weights.IMAGENET1K_V1 if pretrained else None,
            'resnet50': ResNet50_Weights.IMAGENET1K_V1 if pretrained else None,
            'resnet101': ResNet101_Weights.IMAGENET1K_V1 if pretrained else None,
        }
        
        if backbone_name not in weights_map:
            raise ValueError(f"Backbone {backbone_name} not supported")
        
        backbone_fn = getattr(models, backbone_name)
        backbone = backbone_fn(weights=weights_map[backbone_name])
        backbone = nn.Sequential(*list(backbone.children())[:-1])
        return backbone
    
    def _get_3d_backbone(self, backbone_name, pretrained):
        if not MONAI_AVAILABLE:
            raise ImportError("MONAI required for 3D models")
        
        if backbone_name.startswith('resnet'):
            depth = int(backbone_name.replace("resnet", ""))
            monai_resnet_map = {18: monai_nets.resnet18, 34: monai_nets.resnet34,
                               50: monai_nets.resnet50, 101: monai_nets.resnet101}
            
            if depth not in monai_resnet_map:
                raise ValueError(f"3D ResNet depth {depth} not supported")
            
            backbone = monai_resnet_map[depth](pretrained=False, spatial_dims=3, n_input_channels=3)
            backbone.fc = nn.Identity()
            return backbone
        else:
            raise ValueError(f"3D version of {backbone_name} not implemented")
    
    def forward(self, x):
        h = self.encoder(x).flatten(start_dim=1)
        h = self.projection(h)
        logits = self.classifier(h)
        return h, logits
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        _, logits = self(images)

        if self.task == 'multi-label':
            loss = self.criterion(logits, labels.float())
            predicted = (torch.sigmoid(logits) > 0.5).float()
            acc = (predicted == labels).all(dim=1).float().mean()
        else:
            loss = self.criterion(logits, labels.view(-1))
            _, predicted = logits.max(1)
            acc = predicted.eq(labels.view(-1)).float().mean()

        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        _, logits = self(images)

        if self.task == 'multi-label':
            loss = self.criterion(logits, labels.float())
            probs = torch.sigmoid(logits)
            predicted = (probs > 0.5).float()
            self.validation_step_outputs.append({
                'loss': loss, 'probs': probs, 'preds': predicted, 'labels': labels
            })
        else:
            loss = self.criterion(logits, labels.view(-1))
            probs = F.softmax(logits, dim=1)
            _, predicted = logits.max(1)
            self.validation_step_outputs.append({
                'loss': loss, 'probs': probs, 'preds': predicted, 'labels': labels.view(-1)
            })

        return loss
    
    def on_validation_epoch_end(self):
        all_probs = torch.cat([x['probs'] for x in self.validation_step_outputs])
        all_preds = torch.cat([x['preds'] for x in self.validation_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs])
        y_true = all_labels.cpu().numpy()
        y_prob = all_probs.cpu().numpy()

        if self.task == 'multi-label':
            # Multi-label: exact match accuracy
            acc = all_preds.eq(all_labels).all(dim=1).float().mean()
            # Multi-label AUC
            try:
                val_auc = roc_auc_score(y_true, y_prob, average=self.average)
            except:
                val_auc = 0.0
        else:
            # Multi-class: standard accuracy
            acc = all_preds.eq(all_labels).float().mean()
            # Multi-class AUC
            if self.num_classes == 2:
                y_prob_pos = y_prob[:, 1]
                try:
                    val_auc = roc_auc_score(y_true, y_prob_pos)
                except:
                    val_auc = 0.0
            else:
                try:
                    classes_present = sorted(set(y_true))
                    y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
                    if len(classes_present) < self.num_classes:
                        val_auc = roc_auc_score(y_true_bin[:, classes_present], y_prob[:, classes_present], average=self.average)
                    else:
                        val_auc = roc_auc_score(y_true_bin, y_prob, average=self.average)
                except:
                    val_auc = 0.0
        
        self.log('val_loss', torch.stack([x['loss'] for x in self.validation_step_outputs]).mean(), prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_auc', val_auc, prog_bar=True)
        self.validation_step_outputs.clear()
    
    def test_step(self, batch, batch_idx):
        images, labels = batch
        _, logits = self(images)

        if self.task == 'multi-label':
            loss = self.criterion(logits, labels.float())
            probs = torch.sigmoid(logits)
            predicted = (probs > 0.5).float()
            self.test_step_outputs.append({
                'loss': loss, 'probs': probs, 'preds': predicted, 'labels': labels
            })
        else:
            loss = self.criterion(logits, labels.view(-1))
            probs = F.softmax(logits, dim=1)
            _, predicted = logits.max(1)
            self.test_step_outputs.append({
                'loss': loss, 'probs': probs, 'preds': predicted, 'labels': labels.view(-1)
            })

        return loss
    
    def on_test_epoch_end(self):
        all_probs = torch.cat([x['probs'] for x in self.test_step_outputs])
        all_preds = torch.cat([x['preds'] for x in self.test_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.test_step_outputs])
        acc = all_preds.eq(all_labels).float().mean()
        
        y_true = all_labels.cpu().numpy()
        y_prob = all_probs.cpu().numpy()
        
        if self.num_classes == 2:
            y_prob_pos = y_prob[:, 1]
            try:
                test_auc = roc_auc_score(y_true, y_prob_pos)
            except:
                test_auc = 0.0
        else:
            try:
                classes_present = sorted(set(y_true))
                y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
                if len(classes_present) < self.num_classes:
                    test_auc = roc_auc_score(y_true_bin[:, classes_present], y_prob[:, classes_present], average=self.average)
                else:
                    test_auc = roc_auc_score(y_true_bin, y_prob, average=self.average)
            except:
                test_auc = 0.0
        
        self.log('test_loss', torch.stack([x['loss'] for x in self.test_step_outputs]).mean())
        self.log('test_acc', acc)
        self.log('test_auc', test_auc)
        self.test_step_outputs.clear()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate,
                                    weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs, eta_min=0)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}}
    
    def get_predictions(self, dataloader):
        self.eval()
        all_probs, all_preds, all_labels = [], [], []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                _, logits = self(images)
                probs = F.softmax(logits, dim=1)
                _, predicted = logits.max(1)
                all_probs.append(probs.cpu())
                all_preds.append(predicted.cpu())
                all_labels.append(labels.cpu())
        
        all_probs = torch.cat(all_probs)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        y_true = all_labels.numpy().flatten()
        y_pred = all_preds.numpy()
        y_prob = all_probs[:, 1].numpy() if self.num_classes == 2 else all_probs.numpy()
        
        return y_true, y_pred, y_prob


class MLPClassifier(pl.LightningModule):
    """MLP classifier for downstream task."""
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: list = [512, 256],
                 lr: float = 0.001, weight_decay: float = 1e-4, average: str = 'macro', task: str = 'multi-class',
                 max_epochs: int = 100):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.average = average
        self.task = task
        self.max_epochs = max_epochs
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.5)])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

        # Use appropriate loss function based on task type
        if task == 'multi-label':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.validation_step_outputs = []
        self.test_step_outputs = []
    
    def forward(self, x):
        return self.classifier(x)
    
    def training_step(self, batch, batch_idx):
        features, labels = batch
        outputs = self(features)

        if self.task == 'multi-label':
            loss = self.criterion(outputs, labels.float())
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            acc = (predicted == labels).all(dim=1).float().mean()
        else:
            loss = self.criterion(outputs, labels.view(-1))
            _, predicted = outputs.max(1)
            acc = predicted.eq(labels.view(-1)).float().mean()

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        features, labels = batch
        outputs = self(features)

        if self.task == 'multi-label':
            loss = self.criterion(outputs, labels.float())
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()
            self.validation_step_outputs.append({
                'loss': loss, 'probs': probs, 'preds': predicted, 'labels': labels
            })
        else:
            loss = self.criterion(outputs, labels.view(-1))
            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            self.validation_step_outputs.append({
                'loss': loss, 'probs': probs, 'preds': predicted, 'labels': labels.view(-1)
            })

        return loss
    
    def on_validation_epoch_end(self):
        all_probs = torch.cat([x['probs'] for x in self.validation_step_outputs])
        all_preds = torch.cat([x['preds'] for x in self.validation_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs])
        y_true = all_labels.cpu().numpy()
        y_prob = all_probs.cpu().numpy()

        if self.task == 'multi-label':
            # Multi-label: exact match accuracy
            acc = all_preds.eq(all_labels).all(dim=1).float().mean()
            # Multi-label AUC
            try:
                val_auc = roc_auc_score(y_true, y_prob, average=self.average)
            except:
                val_auc = 0.0
        else:
            # Multi-class: standard accuracy
            acc = all_preds.eq(all_labels).float().mean()
            # Multi-class AUC
            if self.num_classes == 2:
                y_prob_pos = y_prob[:, 1]
                try:
                    val_auc = roc_auc_score(y_true, y_prob_pos)
                except:
                    val_auc = 0.0
            else:
                try:
                    classes_present = sorted(set(y_true))
                    y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
                    if len(classes_present) < self.num_classes:
                        val_auc = roc_auc_score(y_true_bin[:, classes_present], y_prob[:, classes_present], average=self.average)
                    else:
                        val_auc = roc_auc_score(y_true_bin, y_prob, average=self.average)
                except:
                    val_auc = 0.0
        
        self.log('val_loss', torch.stack([x['loss'] for x in self.validation_step_outputs]).mean(), prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_auc', val_auc, prog_bar=True)
        self.validation_step_outputs.clear()
    
    def test_step(self, batch, batch_idx):
        features, labels = batch
        outputs = self(features)
        loss = self.criterion(outputs, labels.view(-1))
        probs = F.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        self.test_step_outputs.append({
            'loss': loss, 'probs': probs, 'preds': predicted, 'labels': labels.view(-1)
        })
        return loss
    
    def on_test_epoch_end(self):
        all_probs = torch.cat([x['probs'] for x in self.test_step_outputs])
        all_preds = torch.cat([x['preds'] for x in self.test_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.test_step_outputs])
        acc = all_preds.eq(all_labels).float().mean()
        
        y_true = all_labels.cpu().numpy()
        y_prob = all_probs.cpu().numpy()
        
        if self.num_classes == 2:
            y_prob_pos = y_prob[:, 1]
            try:
                test_auc = roc_auc_score(y_true, y_prob_pos)
            except:
                test_auc = 0.0
        else:
            try:
                classes_present = sorted(set(y_true))
                y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
                if len(classes_present) < self.num_classes:
                    test_auc = roc_auc_score(y_true_bin[:, classes_present], y_prob[:, classes_present], average=self.average)
                else:
                    test_auc = roc_auc_score(y_true_bin, y_prob, average=self.average)
            except:
                test_auc = 0.0
        
        self.log('test_loss', torch.stack([x['loss'] for x in self.test_step_outputs]).mean())
        self.log('test_acc', acc)
        self.log('test_auc', test_auc)
        self.test_step_outputs.clear()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=0)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}}
    
    def get_predictions(self, dataloader, device):
        self.eval()
        all_probs, all_preds, all_labels = [], [], []
        
        with torch.no_grad():
            for batch in dataloader:
                features, labels = batch
                features = features.to(device)
                outputs = self(features)
                probs = F.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                all_probs.append(probs.cpu())
                all_preds.append(predicted.cpu())
                all_labels.append(labels.cpu())
        
        all_probs = torch.cat(all_probs)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        y_true = all_labels.numpy().flatten()
        y_pred = all_preds.numpy()
        y_prob = all_probs[:, 1].numpy() if self.num_classes == 2 else all_probs.numpy()
        
        return y_true, y_pred, y_prob


class LearnableWeightedClassifier(pl.LightningModule):
    """MLP classifier with learnable augmentation group weights."""

    def __init__(self, shared_dim: int, specific_dim: int, num_groups: int, num_classes: int,
                 hidden_dims: list = [512, 256], lr: float = 0.001, weight_decay: float = 1e-4,
                 average: str = 'macro', task: str = 'multi-class', max_epochs: int = 100,
                 weight_init_mode: str = 'uniform', darts_weights: dict = None):
        """
        Args:
            shared_dim: Dimension of shared space (e.g., 128)
            specific_dim: Dimension of each specific group (e.g., 128)
            num_groups: Number of specific augmentation groups (e.g., 5)
            num_classes: Number of output classes
            hidden_dims: Hidden layer dimensions
            lr: Learning rate
            weight_decay: Weight decay
            average: 'macro' or 'micro' for multi-class metrics
            task: 'multi-class' or 'multi-label'
            max_epochs: Max training epochs
            weight_init_mode: 'uniform', 'darts', or 'none'
            darts_weights: Dictionary of DARTS weights (required if weight_init_mode='darts')
        """
        super().__init__()
        self.save_hyperparameters()

        self.shared_dim = shared_dim
        self.specific_dim = specific_dim
        self.num_groups = num_groups
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.average = average
        self.task = task
        self.max_epochs = max_epochs
        self.weight_init_mode = weight_init_mode

        # Initialize learnable weights based on mode
        if weight_init_mode == 'uniform':
            # Uniform initialization (all groups equal)
            init_weights = torch.ones(num_groups) / num_groups
        elif weight_init_mode == 'darts':
            # Initialize from DARTS weights
            if darts_weights is None:
                raise ValueError("darts_weights must be provided when weight_init_mode='darts'")
            aug_names = ['rotation', 'translation', 'scaling', 'contrast', 'noise']
            init_weights = torch.tensor([darts_weights[name] for name in aug_names])
        elif weight_init_mode == 'none':
            # No weighting (all weights = 1.0, no softmax)
            init_weights = torch.ones(num_groups)
        else:
            raise ValueError(f"Invalid weight_init_mode: {weight_init_mode}")

        # Store as learnable parameter (in log-space for better optimization)
        if weight_init_mode == 'none':
            # No learning, just fixed weights
            self.register_buffer('alphas', init_weights)
            self.learnable_weights = False
        else:
            # Learnable weights in log-space
            self.alphas = nn.Parameter(torch.log(init_weights + 1e-8))
            self.learnable_weights = True

        # MLP classifier
        input_dim = shared_dim + num_groups * specific_dim
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.5)])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

        # Loss function
        if task == 'multi-label':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, features):
        """
        Args:
            features: [batch, shared_dim + num_groups * specific_dim]
                     e.g., [batch, 128 + 5*128] = [batch, 768]
        """
        batch_size = features.size(0)

        # Split into shared and specific features
        shared = features[:, :self.shared_dim]  # [batch, 128]
        specific = features[:, self.shared_dim:].reshape(batch_size, self.num_groups, self.specific_dim)  # [batch, 5, 128]

        # Apply weights
        if self.learnable_weights:
            weights = F.softmax(self.alphas, dim=0)  # [5]
        else:
            weights = self.alphas  # No softmax for 'none' mode

        weighted_specific = (specific * weights.view(1, self.num_groups, 1)).reshape(batch_size, -1)  # [batch, 640]

        # Concatenate shared and weighted specific
        combined = torch.cat([shared, weighted_specific], dim=1)  # [batch, 768]

        return self.classifier(combined)

    def training_step(self, batch, batch_idx):
        features, labels = batch
        outputs = self(features)

        if self.task == 'multi-label':
            loss = self.criterion(outputs, labels.float())
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            acc = (predicted == labels).all(dim=1).float().mean()
        else:
            loss = self.criterion(outputs, labels.view(-1))
            _, predicted = outputs.max(1)
            acc = predicted.eq(labels.view(-1)).float().mean()

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        outputs = self(features)

        if self.task == 'multi-label':
            loss = self.criterion(outputs, labels.float())
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            probs = torch.sigmoid(outputs)
        else:
            loss = self.criterion(outputs, labels.view(-1))
            _, predicted = outputs.max(1)
            probs = F.softmax(outputs, dim=1)

        self.validation_step_outputs.append({
            'loss': loss,
            'preds': predicted,
            'probs': probs,
            'labels': labels
        })

        return loss

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        all_preds = torch.cat([x['preds'] for x in self.validation_step_outputs])
        all_probs = torch.cat([x['probs'] for x in self.validation_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs])

        if self.task == 'multi-label':
            acc = (all_preds == all_labels).all(dim=1).float().mean()
        else:
            acc = all_preds.eq(all_labels).float().mean()

        y_true = all_labels.cpu().numpy()
        y_prob = all_probs.cpu().numpy()

        if self.num_classes == 2:
            y_prob_pos = y_prob[:, 1]
            try:
                val_auc = roc_auc_score(y_true, y_prob_pos)
            except:
                val_auc = 0.0
        else:
            try:
                classes_present = sorted(set(y_true))
                y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
                if len(classes_present) < self.num_classes:
                    val_auc = roc_auc_score(y_true_bin[:, classes_present], y_prob[:, classes_present], average=self.average)
                else:
                    val_auc = roc_auc_score(y_true_bin, y_prob, average=self.average)
            except:
                val_auc = 0.0

        self.log('val_loss', avg_loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_auc', val_auc, prog_bar=True)

        # Log current weights if learnable
        if self.learnable_weights:
            weights = F.softmax(self.alphas, dim=0).detach().cpu().numpy()
            aug_names = ['rotation', 'translation', 'scaling', 'contrast', 'noise']
            for i, name in enumerate(aug_names):
                self.log(f'weight_{name}', weights[i], prog_bar=False)

        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        features, labels = batch
        outputs = self(features)

        if self.task == 'multi-label':
            loss = self.criterion(outputs, labels.float())
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            probs = torch.sigmoid(outputs)
        else:
            loss = self.criterion(outputs, labels.view(-1))
            _, predicted = outputs.max(1)
            probs = F.softmax(outputs, dim=1)

        self.test_step_outputs.append({
            'loss': loss,
            'preds': predicted,
            'probs': probs,
            'labels': labels.view(-1) if self.task != 'multi-label' else labels
        })

        return loss

    def on_test_epoch_end(self):
        all_preds = torch.cat([x['preds'] for x in self.test_step_outputs])
        all_probs = torch.cat([x['probs'] for x in self.test_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.test_step_outputs])

        if self.task == 'multi-label':
            acc = (all_preds == all_labels).all(dim=1).float().mean()
        else:
            acc = all_preds.eq(all_labels).float().mean()

        y_true = all_labels.cpu().numpy()
        y_prob = all_probs.cpu().numpy()

        if self.num_classes == 2:
            y_prob_pos = y_prob[:, 1]
            try:
                test_auc = roc_auc_score(y_true, y_prob_pos)
            except:
                test_auc = 0.0
        else:
            try:
                classes_present = sorted(set(y_true))
                y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
                if len(classes_present) < self.num_classes:
                    test_auc = roc_auc_score(y_true_bin[:, classes_present], y_prob[:, classes_present], average=self.average)
                else:
                    test_auc = roc_auc_score(y_true_bin, y_prob, average=self.average)
            except:
                test_auc = 0.0

        self.log('test_loss', torch.stack([x['loss'] for x in self.test_step_outputs]).mean())
        self.log('test_acc', acc)
        self.log('test_auc', test_auc)
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=0)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}}

    def get_predictions(self, dataloader, device):
        self.eval()
        all_probs, all_preds, all_labels = [], [], []

        with torch.no_grad():
            for batch in dataloader:
                features, labels = batch
                features = features.to(device)
                outputs = self(features)
                probs = F.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                all_probs.append(probs.cpu())
                all_preds.append(predicted.cpu())
                all_labels.append(labels.cpu())

        all_probs = torch.cat(all_probs)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        y_true = all_labels.numpy().flatten()
        y_pred = all_preds.numpy()
        y_prob = all_probs[:, 1].numpy() if self.num_classes == 2 else all_probs.numpy()

        return y_true, y_pred, y_prob

    def get_final_weights(self):
        """Get final learned weights."""
        if self.learnable_weights:
            weights = F.softmax(self.alphas, dim=0).detach().cpu().numpy()
        else:
            weights = self.alphas.cpu().numpy()

        aug_names = ['rotation', 'translation', 'scaling', 'contrast', 'noise']
        return {name: float(weight) for name, weight in zip(aug_names, weights)}


def extract_group_features(z, aug_groups, group_size, num_aug_groups=6, include_shared=True):
    """Extract features from specific augmentation groups.

    With soft disentanglement:
    - 5 specific groups: rotation, translation, scaling, contrast, noise
    - 1 shared space (implicit, optionally included)
    - Total projection_dim = 768 = 5 × 128 + 128

    Args:
        z: Full feature vector [batch, 768]
        aug_groups: List of augmentation groups to extract
        group_size: Size of each group (e.g., 128)
        num_aug_groups: Total number of groups including shared (e.g., 6)
        include_shared: Whether to include shared space (default: True for backward compatibility)
    """
    # Mapping: only 5 specific groups (shared space is separate)
    aug_to_idx = {'rotation': 0, 'translation': 1, 'scaling': 2, 'contrast': 3, 'noise': 4}

    group_indices = []
    for group in aug_groups:
        if isinstance(group, str):
            if group in aug_to_idx:
                group_indices.append(aug_to_idx[group])
            else:
                raise ValueError(f"Unknown augmentation group: {group}. Valid: {list(aug_to_idx.keys())}")
        else:
            group_indices.append(int(group))

    # Extract specific group dimensions
    features_list = []
    for idx in group_indices:
        start_idx = idx * group_size
        end_idx = (idx + 1) * group_size
        features_list.append(z[:, start_idx:end_idx])

    # Optionally include shared space (last dimensions)
    if include_shared:
        num_specific_groups = num_aug_groups - 1  # 5
        total_specific_dims = num_specific_groups * group_size  # 640
        shared_features = z[:, total_specific_dims:]  # 640:768
        features_list.append(shared_features)

    return torch.cat(features_list, dim=1)


def extract_features(encoder, dataloader, device, aug_groups=None, group_size=None, use_disentanglement=False, num_aug_groups=6, include_shared=True):
    """Extract features from the frozen encoder.

    Args:
        include_shared: Whether to include shared space when extracting group features (default: True)
    """
    encoder.eval()
    all_features, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features", leave=False):
            images, labels = batch
            images = images.to(device)

            # All methods return (h, z, ...) - we only need z (concatenated groups)
            output = encoder(images)
            z = output[1]

            if use_disentanglement and aug_groups is not None:
                z = extract_group_features(z, aug_groups, group_size, num_aug_groups, include_shared=include_shared)

            all_features.append(z.cpu())
            all_labels.append(labels.cpu() if isinstance(labels, torch.Tensor) else torch.tensor(labels))

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return features, labels


def train_classifier(train_features, train_labels, val_features, val_labels,
                     num_classes, device, num_epochs=100, lr=0.001,
                     use_wandb=False, average='macro', output_dir=None, task='multi-class',
                     early_stopping=False, early_stopping_patience=50):
    """Train MLP classifier on extracted features."""
    input_dim = train_features.shape[1]

    model = MLPClassifier(input_dim=input_dim, num_classes=num_classes,
                         lr=lr, weight_decay=1e-4, average=average, task=task, max_epochs=num_epochs)

    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

    callbacks = []

    if output_dir:
        checkpoint_callback = ModelCheckpoint(
            dirpath=output_dir, filename='best_classifier',
            monitor='val_auc', mode='max', save_top_k=1, save_last=False, verbose=False
        )
        callbacks.append(checkpoint_callback)

    if early_stopping:
        early_stop_callback = EarlyStopping(monitor='val_auc', patience=early_stopping_patience, mode='max', verbose=False)
        callbacks.append(early_stop_callback)
    
    logger = None
    if use_wandb:
        if wandb.run is not None:
            logger = False
        else:
            from pytorch_lightning.loggers import WandbLogger
            logger = WandbLogger(project='simclr-classifier', log_model=False)
    
    trainer = pl.Trainer(
        max_epochs=num_epochs, accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1, callbacks=callbacks, logger=logger, enable_progress_bar=True,
        enable_model_summary=False, log_every_n_steps=10, 
        enable_checkpointing=True if output_dir else False
    )
    
    trainer.fit(model, train_loader, val_loader)
    
    if output_dir and checkpoint_callback.best_model_path:
        model = MLPClassifier.load_from_checkpoint(checkpoint_callback.best_model_path)
        best_val_auc = checkpoint_callback.best_model_score.item()
    else:
        best_val_auc = trainer.callback_metrics.get('val_auc', 0.0)
        if isinstance(best_val_auc, torch.Tensor):
            best_val_auc = best_val_auc.item()
    
    return model, best_val_auc


def differentiable_group_selection(encoder, train_loader, val_loader, num_classes,
                                   group_size, num_groups, device, num_epochs=100,
                                   use_wandb=False, average='micro', output_dir=None, task='multi-class',
                                   early_stopping=False, early_stopping_patience=50,
                                   darts_type='first_order', clf_lr=0.001, arch_lr=0.01,
                                   clf_weight_decay=1e-4, arch_weight_decay=0.0,
                                   clf_eta_min=1e-5, batch_size=128, discretization_method='topk',
                                   classifier_epochs=100):
    """DARTS-inspired group selection.

    Two modes based on discretization_method:
    - 'topk': Uses CONCATENATION during training. Learns which groups to include/exclude.
              Classifier input = shared(128) + all_aug_groups(5*128) = 768 dims
              Architecture weights act as soft gates on each group's contribution.
    - 'weighted': Uses WEIGHTED SUM during training. Learns relative importance of groups.
                  Classifier input = shared(128) + weighted_sum(128) = 256 dims
                  Architecture weights determine contribution to weighted average.
    """
    print("\n" + "="*70)
    print("DIFFERENTIABLE GROUP SELECTION (DARTS)")
    print("="*70)

    # Only 5 specific augmentation groups (shared space is implicit, always included)
    aug_names = ['rotation', 'translation', 'scaling', 'contrast', 'noise']
    num_specific_groups = num_groups - 1  # 5 specific groups

    print("\nExtracting features for all groups...")
    all_group_features = {}

    # Extract labels once (they're the same for all groups)
    train_labels = None
    val_labels = None

    print(f"\nExtracting each group separately (without shared space)...")

    for group_idx in range(num_specific_groups):
        train_feats, train_labels_temp = extract_features(
            encoder, train_loader, device,
            aug_groups=[group_idx], group_size=group_size, use_disentanglement=True,
            num_aug_groups=num_groups, include_shared=False
        )
        val_feats, val_labels_temp = extract_features(
            encoder, val_loader, device,
            aug_groups=[group_idx], group_size=group_size, use_disentanglement=True,
            num_aug_groups=num_groups, include_shared=False
        )
        all_group_features[group_idx] = {'train': train_feats, 'val': val_feats}

        # Store labels from first iteration
        if train_labels is None:
            train_labels = train_labels_temp
            val_labels = val_labels_temp

    # Extract shared space ONCE separately
    print(f"Extracting shared space (once)...")
    total_specific_dims = num_specific_groups * group_size  # 640

    shared_train_feats = []
    with torch.no_grad():
        encoder.eval()
        for batch in tqdm(train_loader, desc="Extracting shared (train)", leave=False):
            images, _ = batch
            images = images.to(device)
            output = encoder(images)
            z = output[1]
            shared = z[:, total_specific_dims:]  # [batch, 128] - shared/semantic space only
            shared_train_feats.append(shared.cpu())
    shared_train = torch.cat(shared_train_feats, dim=0)

    shared_val_feats = []
    with torch.no_grad():
        encoder.eval()
        for batch in tqdm(val_loader, desc="Extracting shared (val)", leave=False):
            images, _ = batch
            images = images.to(device)
            output = encoder(images)
            z = output[1]
            shared = z[:, total_specific_dims:]  # [batch, 128] - shared/semantic space only
            shared_val_feats.append(shared.cpu())
    shared_val = torch.cat(shared_val_feats, dim=0)

    # Architecture parameters only for specific groups (not shared space)
    arch_params = nn.Parameter(torch.ones(num_specific_groups, device=device) / num_specific_groups)

    # Feature dim per group is now ONLY specific dims (no shared space duplication)
    feature_dim_per_group = all_group_features[0]['train'].shape[1]  # Should be group_size (128)
    shared_dim = shared_train.shape[1]  # 128

    # Classifier input dimension depends on discretization method
    if discretization_method == 'topk':
        # Concatenation: shared + all aug groups (each scaled by weight)
        classifier_input_dim = shared_dim + feature_dim_per_group * num_specific_groups
        print(f"\nDARTS Mode: CONCATENATION (for top-k selection)")
        print(f"  Each aug group is scaled by its softmax weight, then concatenated")
        print(f"  Classifier input: {shared_dim} (shared) + {feature_dim_per_group}*{num_specific_groups} (aug groups) = {classifier_input_dim} dims")
    else:
        # Weighted sum: shared + weighted combination of aug groups
        classifier_input_dim = shared_dim + feature_dim_per_group
        print(f"\nDARTS Mode: WEIGHTED SUM (for weighted combination)")
        print(f"  Aug groups are combined via weighted sum into single vector")
        print(f"  Classifier input: {shared_dim} (shared) + {feature_dim_per_group} (weighted_sum) = {classifier_input_dim} dims")

    print(f"\nFeature dimensions:")
    print(f"  Per specific group: {feature_dim_per_group}")
    print(f"  Shared space: {shared_dim}")
    print(f"  Classifier input: {classifier_input_dim}")

    temp_classifier = nn.Sequential(
        nn.Linear(classifier_input_dim, 512), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    ).to(device)

    # Create optimizers with provided hyperparameters
    clf_optimizer = torch.optim.Adam(
        temp_classifier.parameters(),
        lr=clf_lr,
        weight_decay=clf_weight_decay
    )
    arch_optimizer = torch.optim.Adam(
        [arch_params],
        lr=arch_lr,
        weight_decay=arch_weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    # Classifier scheduler only (no architecture scheduler)
    clf_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        clf_optimizer,
        T_max=num_epochs,
        eta_min=clf_eta_min
    )

    # Prepare datasets for batch-level alternation (specific groups only)
    train_dataset = TensorDataset(
        torch.cat([all_group_features[i]['train'] for i in range(num_specific_groups)], dim=1),
        shared_train,  # Add shared space as separate tensor
        train_labels
    )
    val_dataset = TensorDataset(
        torch.cat([all_group_features[i]['val'] for i in range(num_specific_groups)], dim=1),
        shared_val,  # Add shared space as separate tensor
        val_labels
    )

    train_loader_clf = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader_clf = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    print(f"\n{'='*70}")
    print(f"DARTS Configuration:")
    print(f"  Type: {darts_type}")
    print(f"  Discretization: {discretization_method}")
    print(f"  Classifier LR: {clf_lr} → {clf_eta_min} (CosineAnnealingLR)")
    print(f"  Architecture LR: {arch_lr} (constant, no scheduler)")
    print(f"  Classifier weight decay: {clf_weight_decay}")
    print(f"  Architecture weight decay: {arch_weight_decay}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"{'='*70}\n")

    best_arch_auc = 0.0  # For monitoring only, not used for selection

    def compute_combined_features(aug_batch, shared_batch, weights, method):
        """Compute combined features based on discretization method."""
        # Split concatenated features into individual group features
        group_features = []
        for i in range(num_specific_groups):
            start_idx = i * feature_dim_per_group
            end_idx = (i + 1) * feature_dim_per_group
            group_features.append(aug_batch[:, start_idx:end_idx])

        if method == 'topk':
            # Concatenation: scale each group by its weight, then concatenate all
            scaled_groups = [w * f for w, f in zip(weights, group_features)]
            combined = torch.cat([shared_batch] + scaled_groups, dim=1)
        else:
            # Weighted sum: combine all groups into single vector
            weighted_sum = sum(w * f for w, f in zip(weights, group_features))
            combined = torch.cat([shared_batch, weighted_sum], dim=1)

        return combined

    # Batch-level alternating optimization
    for epoch in range(num_epochs):
        temp_classifier.train()

        # Create iterators for batch-level alternation
        train_iter = iter(train_loader_clf)
        val_iter = iter(val_loader_clf)

        num_batches = len(train_loader_clf)

        for batch_idx in range(num_batches):
            # ========================================
            # Step 1: Update classifier weights on TRAIN batch
            # ========================================
            try:
                train_aug_batch, train_shared_batch, train_labels_batch = next(train_iter)
            except StopIteration:
                break

            train_aug_batch = train_aug_batch.to(device)
            train_shared_batch = train_shared_batch.to(device)
            train_labels_batch = train_labels_batch.to(device)

            weights = F.softmax(arch_params, dim=0).detach()  # Don't track for classifier update
            combined_train = compute_combined_features(train_aug_batch, train_shared_batch, weights, discretization_method)

            clf_optimizer.zero_grad()
            train_outputs = temp_classifier(combined_train)
            train_loss = criterion(train_outputs, train_labels_batch.view(-1))
            train_loss.backward()
            clf_optimizer.step()

            # ========================================
            # Step 2: Update architecture on VAL batch
            # ========================================
            try:
                val_aug_batch, val_shared_batch, val_labels_batch = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader_clf)
                val_aug_batch, val_shared_batch, val_labels_batch = next(val_iter)

            val_aug_batch = val_aug_batch.to(device)
            val_shared_batch = val_shared_batch.to(device)
            val_labels_batch = val_labels_batch.to(device)

            if darts_type == 'second_order':
                # ========================================
                # Second-Order DARTS: Use virtual weights
                # ========================================
                # Save current classifier state
                original_params = [p.clone() for p in temp_classifier.parameters()]

                # Compute gradient on train batch with differentiable weights
                weights = F.softmax(arch_params, dim=0)
                combined_train_virt = compute_combined_features(train_aug_batch, train_shared_batch, weights, discretization_method)

                outputs_virt = temp_classifier(combined_train_virt)
                loss_virt = criterion(outputs_virt, train_labels_batch.view(-1))

                # Compute gradients w.r.t. classifier parameters
                grads = torch.autograd.grad(loss_virt, temp_classifier.parameters(), create_graph=True)

                # Virtual step: w' = w - lr * grad
                with torch.no_grad():
                    for p, g in zip(temp_classifier.parameters(), grads):
                        p.data = p.data - clf_lr * g.data

                # Now compute architecture loss using virtual weights
                weights = F.softmax(arch_params, dim=0)
                combined_val = compute_combined_features(val_aug_batch, val_shared_batch, weights, discretization_method)

                val_outputs = temp_classifier(combined_val)
                val_loss = criterion(val_outputs, val_labels_batch.view(-1))

                # Update architecture parameters
                arch_optimizer.zero_grad()
                val_loss.backward()
                arch_optimizer.step()

                # Restore original classifier parameters
                with torch.no_grad():
                    for p, p_orig in zip(temp_classifier.parameters(), original_params):
                        p.data = p_orig.data

            else:
                # ========================================
                # First-Order DARTS: Use current weights
                # ========================================
                weights = F.softmax(arch_params, dim=0)
                combined_val = compute_combined_features(val_aug_batch, val_shared_batch, weights, discretization_method)

                val_outputs = temp_classifier(combined_val)
                val_loss = criterion(val_outputs, val_labels_batch.view(-1))

                # Update architecture parameters
                arch_optimizer.zero_grad()
                val_loss.backward()
                arch_optimizer.step()

        # ========================================
        # End of epoch: Evaluate on full validation set
        # ========================================
        temp_classifier.eval()
        with torch.no_grad():
            val_aug_full = torch.cat([all_group_features[i]['val'] for i in range(num_specific_groups)], dim=1).to(device)
            val_shared_full = shared_val.to(device)
            val_labels_full = val_labels.to(device)

            weights = F.softmax(arch_params, dim=0)
            combined_val_full = compute_combined_features(val_aug_full, val_shared_full, weights, discretization_method)

            val_outputs_full = temp_classifier(combined_val_full)
            val_loss_full = criterion(val_outputs_full, val_labels_full.view(-1))

            val_probs = F.softmax(val_outputs_full, dim=1)
            _, val_predicted = val_outputs_full.max(1)
            val_acc = 100. * val_predicted.eq(val_labels_full.view(-1)).sum().item() / val_labels_full.size(0)

            y_true_val = val_labels_full.cpu().numpy().flatten()
            y_prob_val = val_probs.cpu().numpy()

            if num_classes == 2:
                y_prob_pos = y_prob_val[:, 1]
                try:
                    val_auc = roc_auc_score(y_true_val, y_prob_pos)
                except:
                    val_auc = 0.0
            else:
                try:
                    classes_present = sorted(set(y_true_val))
                    y_true_bin = label_binarize(y_true_val, classes=range(num_classes))
                    if len(classes_present) < num_classes:
                        val_auc = roc_auc_score(y_true_bin[:, classes_present], y_prob_val[:, classes_present], average=average)
                    else:
                        val_auc = roc_auc_score(y_true_bin, y_prob_val, average=average)
                except:
                    val_auc = 0.0

        if val_auc > best_arch_auc:
            best_arch_auc = val_auc

        weights_np = F.softmax(arch_params, dim=0).detach().cpu().numpy()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Val Acc: {val_acc:.2f}%, Val AUC: {val_auc:.4f} (Best: {best_arch_auc:.4f})")
            print(f"  Weights: {' '.join([f'{aug_names[i]}={weights_np[i]:.3f}' for i in range(num_specific_groups)])}")
            print(f"  Classifier LR: {clf_optimizer.param_groups[0]['lr']:.6f}")

        if use_wandb:
            log_dict = {
                'darts_val_acc': val_acc,
                'darts_val_auc': val_auc,
                'darts_val_loss': val_loss_full.item(),
                'darts_best_auc': best_arch_auc,
                'darts_epoch': epoch,
                'darts_clf_lr': clf_optimizer.param_groups[0]['lr']
            }
            for i in range(num_specific_groups):
                log_dict[f'darts_weight_{aug_names[i]}'] = weights_np[i]
            wandb.log(log_dict)

        # Step classifier scheduler only (no architecture scheduler)
        clf_scheduler.step()

    # Use FINAL weights (standard DARTS practice), not best AUC weights
    final_weights = F.softmax(arch_params, dim=0).detach().cpu().numpy()

    print("\n" + "="*70)
    print("FINAL AUGMENTATION GROUP WEIGHTS (after convergence)")
    print("="*70)
    print(f"(Best validation AUC during training: {best_arch_auc:.4f})")
    for i in range(num_specific_groups):
        print(f"  {aug_names[i]:10s}: {final_weights[i]:.4f}")

    # Discretization based on method
    weights_dict = {aug_names[i]: float(final_weights[i]) for i in range(num_specific_groups)}
    os.environ['PYTORCH_LIGHTNING_VERBOSITY'] = '0'

    if discretization_method == 'weighted':
        # Weighted concatenation: use all groups with sqrt(weight) scaling
        print("\nDiscretization method: Weighted concatenation (soft selection)")
        print("Skipping intermediate classifier training - will apply weights directly in final classifier")
        print("DARTS learned weights:")

        for group_idx in range(num_specific_groups):
            weight = final_weights[group_idx]
            scale_factor = np.sqrt(weight)
            print(f"  {aug_names[group_idx]:10s}: weight={weight:.4f}, scale={scale_factor:.4f}")

        print(f"\n✓ Feature dimension: {group_size} (shared) + {group_size} (weighted aug) = {group_size * 2}")
        print("="*70 + "\n")

        # Return all groups (indices) and weights for weighted method
        # No intermediate AUC since we skip discretization classifier training
        selected_groups = list(range(num_specific_groups))
        return selected_groups, None, weights_dict

    else:
        # Top-k selection: hard selection of top groups
        print("\nDiscretization method: Top-k (hard selection)")
        strategies = [("Shared", None), ("Top-1", 1), ("Top-2", 2), ("Top-3", 3), ("Top-4", 4), ("All", num_specific_groups)]
        best_strategy = None
        best_auc = 0.0
        best_classifier = None
        best_train_feats = None
        best_val_feats = None

        print(f"Evaluating discretization strategies (training {classifier_epochs} epochs each):")
        print("  Each strategy tests: shared + selected aug groups")

        for strategy_name, k in strategies:
            if strategy_name == "Shared":
                # Special case: use only shared space (no augmentation groups)
                # This tests if aug-specific features add any value
                train_feats_selected = shared_train
                val_feats_selected = shared_val
                selected_groups = ['shared_only']
                top_k_indices = []
            else:
                top_k_indices = np.argsort(final_weights)[-k:][::-1]
                selected_groups = [aug_names[i] for i in top_k_indices]

                # Concatenate shared + selected aug groups
                train_aug_selected = torch.cat([all_group_features[i]['train'] for i in top_k_indices], dim=1)
                val_aug_selected = torch.cat([all_group_features[i]['val'] for i in top_k_indices], dim=1)

                train_feats_selected = torch.cat([shared_train, train_aug_selected], dim=1)
                val_feats_selected = torch.cat([shared_val, val_aug_selected], dim=1)

            clf_temp, strategy_auc = train_classifier(
                train_feats_selected, train_labels, val_feats_selected, val_labels,
                num_classes, device, num_epochs=classifier_epochs, lr=0.001,
                use_wandb=False, average=average, output_dir=output_dir, task=task,
                early_stopping=early_stopping, early_stopping_patience=early_stopping_patience
            )

            if strategy_name == "Shared":
                print(f"  {strategy_name:8s} [shared only]: AUC = {strategy_auc:.4f}")
            else:
                print(f"  {strategy_name:8s} [shared + {selected_groups}]: AUC = {strategy_auc:.4f}")

            if use_wandb:
                wandb.log({
                    f'darts_{strategy_name}_auc': strategy_auc,
                    f'darts_{strategy_name}_groups': ','.join(selected_groups)
                })

            if strategy_auc > best_auc:
                best_auc = strategy_auc
                best_strategy = (strategy_name, top_k_indices)
                best_classifier = clf_temp
                best_train_feats = train_feats_selected
                best_val_feats = val_feats_selected

        # Fallback if no strategy achieved AUC > 0 (e.g., missing classes)
        if best_strategy is None:
            print(f"\nWarning: No strategy achieved AUC > 0, defaulting to All groups")
            best_strategy = ("All", list(range(num_specific_groups)))
            best_classifier = clf_temp
            best_train_feats = train_feats_selected
            best_val_feats = val_feats_selected

        print(f"\n✓ Best strategy: {best_strategy[0]}")
        if best_strategy[0] == "Shared":
            print(f"✓ Selected: shared only (aug groups don't help)")
        else:
            print(f"✓ Selected: shared + {[aug_names[i] for i in best_strategy[1]]}")
        print(f"✓ Best AUC: {best_auc:.4f}")
        print("="*70 + "\n")

        # Return selected groups, best AUC, weights dict, best classifier, and features
        return best_strategy[1], best_auc, weights_dict, best_classifier, best_train_feats, best_val_feats, train_labels, val_labels, all_group_features, shared_train, shared_val


def grid_search_selection(encoder, train_loader, val_loader, num_classes,
                          group_size, num_groups, device, max_epochs=50,
                          use_wandb=False, average='micro', output_dir=None, task='multi-class',
                          early_stopping=False, early_stopping_patience=50):
    """Grid search over all group combinations."""
    print("\n" + "="*70)
    print("GRID SEARCH GROUP SELECTION")
    print("="*70)

    # Only 5 specific augmentation groups (shared space is implicit, always included)
    aug_names = ['rotation', 'translation', 'scaling', 'contrast', 'noise']
    num_specific_groups = num_groups - 1  # 5 specific groups
    results = []

    total_combinations = sum([len(list(combinations(range(num_specific_groups), r))) for r in range(1, num_specific_groups + 1)])
    print(f"Testing {total_combinations} combinations (shared space always included)...")

    pbar = tqdm(total=total_combinations, desc="Grid search")

    for r in range(1, num_specific_groups + 1):
        for groups in combinations(range(num_specific_groups), r):
            group_names = [aug_names[i] for i in groups]

            train_features, train_labels = extract_features(
                encoder, train_loader, device,
                aug_groups=list(groups), group_size=group_size, use_disentanglement=True, num_aug_groups=num_groups
            )

            val_features, val_labels = extract_features(
                encoder, val_loader, device,
                aug_groups=list(groups), group_size=group_size, use_disentanglement=True, num_aug_groups=num_groups
            )
            
            classifier, val_auc = train_classifier(
                train_features, train_labels, val_features, val_labels,
                num_classes, device, num_epochs=max_epochs, lr=0.001,
                use_wandb=False, average=average, output_dir=None, task=task,
                early_stopping=early_stopping, early_stopping_patience=early_stopping_patience
            )
            
            results.append({
                'groups': list(groups),
                'group_names': group_names,
                'auc': val_auc,
                'feature_dim': train_features.shape[1]
            })
            
            pbar.set_postfix({'Best AUC': max(results, key=lambda x: x['auc'])['auc']})
            pbar.update(1)
    
    pbar.close()
    
    best_result = max(results, key=lambda x: x['auc'])
    
    print("\n" + "="*70)
    print("GRID SEARCH RESULTS")
    print("="*70)
    print("\nTop 5 combinations:")
    for result in sorted(results, key=lambda x: -x['auc'])[:5]:
        print(f"  {str(result['group_names']):60s}: AUC = {result['auc']:.4f} (dim={result['feature_dim']})")
    
    print(f"\n✓ Best combination: {best_result['group_names']}")
    print(f"✓ Best AUC: {best_result['auc']:.4f}")
    print("="*70 + "\n")
    
    return best_result['groups'], best_result['auc']


def evaluate_and_save(y_true, y_pred, y_prob, num_classes, dataset_info, args,
                      output_dir, best_val_auc=None, darts_weights=None,
                      use_disentanglement=False, selected_groups=None,
                      selected_names=None, feature_dim=None, phase_name=None):
    """Run evaluation, plot results, and save config for a given set of predictions.

    Returns:
        dict with acc, auc, sens, spec, prec, f1
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f'{__name__}_{phase_name or "main"}')
    logger.setLevel(logging.INFO)
    # Clear existing handlers to avoid duplicate logs
    logger.handlers.clear()
    logger.addHandler(logging.StreamHandler())
    log_file = output_dir / 'main_predict.txt'
    logger.addHandler(logging.FileHandler(log_file, mode='w'))

    # Determine actual number of classes present in the evaluation data
    unique_classes = sorted(set(y_true.tolist() if hasattr(y_true, 'tolist') else y_true))
    eval_num_classes = len(unique_classes)

    # Filter class names to match classes present in data
    if isinstance(dataset_info['label'], dict):
        all_class_names = [str(label) for label in dataset_info['label'].values()]
        if eval_num_classes < num_classes:
            eval_class_names = [all_class_names[i] for i in unique_classes if i < len(all_class_names)]
        else:
            eval_class_names = all_class_names
    else:
        eval_class_names = None

    # Filter y_prob to only include columns for classes present in data
    if eval_num_classes < num_classes and y_prob is not None and hasattr(y_prob, 'shape') and len(y_prob.shape) > 1:
        y_prob_eval = y_prob[:, unique_classes] if y_prob.shape[1] > eval_num_classes else y_prob
        # Remap y_true and y_pred to contiguous 0..eval_num_classes-1
        class_map = {c: i for i, c in enumerate(unique_classes)}
        y_true_eval = np.array([class_map[y] for y in (y_true.tolist() if hasattr(y_true, 'tolist') else y_true)])
        y_pred_eval = np.array([class_map.get(y, y) for y in (y_pred.tolist() if hasattr(y_pred, 'tolist') else y_pred)])
    else:
        y_true_eval, y_pred_eval, y_prob_eval = y_true, y_pred, y_prob

    # Plot results
    acc, sens, spec, prec, f1, cm = plot_confusion_matrix(
        y_true_eval, y_pred_eval,
        output_dir / 'confusion_matrix.png',
        eval_num_classes,
        class_names=eval_class_names,
        average=args.average_method
    )

    auc_val = plot_roc_curve(y_true_eval, y_prob_eval, output_dir / 'roc.png', eval_num_classes, average=args.average_method)

    header = f"FINAL TEST RESULTS ({phase_name})" if phase_name else "FINAL TEST RESULTS"
    logger.info("\n" + "="*70)
    logger.info(header)
    logger.info("="*70)
    logger.info(f"Accuracy: {acc:.4f}")
    if eval_num_classes < num_classes:
        logger.info(f"Evaluated on {eval_num_classes}/{num_classes} classes")
    if eval_num_classes == 2:
        logger.info(f"Sensitivity: {sens:.4f}")
        logger.info(f"Specificity: {spec:.4f}")
        logger.info(f"Precision: {prec:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        logger.info(f"AUC: {auc_val:.4f}")
    else:
        logger.info(f"Sensitivity ({args.average_method}-average): {sens:.4f}")
        logger.info(f"Specificity ({args.average_method}-average): {spec:.4f}")
        logger.info(f"Precision ({args.average_method}-average): {prec:.4f}")
        logger.info(f"F1-Score ({args.average_method}-average): {f1:.4f}")
        logger.info(f"AUC ({args.average_method}-average): {auc_val:.4f}")
    logger.info(f"\nConfusion Matrix:\n{cm}")

    if darts_weights is not None:
        logger.info("\n" + "="*70)
        logger.info("DARTS FINAL AUGMENTATION GROUP WEIGHTS")
        logger.info("="*70)
        for group_name, weight in sorted(darts_weights.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {group_name:15s}: {weight:.6f}")
        logger.info("  shared          : always included (implicit)")
        logger.info("="*70)

    # Save results CSV
    if eval_num_classes == 2:
        results_df = pd.DataFrame({'GT': y_true_eval, 'NN': y_pred_eval, 'NN_pred': y_prob_eval})
    else:
        results_dict = {'GT': y_true_eval, 'NN': y_pred_eval}
        for i in range(eval_num_classes):
            results_dict[f'NN_pred_class_{i}'] = y_prob_eval[:, i]
        results_df = pd.DataFrame(results_dict)
    results_df.to_csv(output_dir / 'results.csv', index=False)

    # Save config
    if args.mode in ['contrastive', 'disentangled']:
        config_save = {
            'mode': args.mode,
            'dataset': args.dataset,
            'use_disentanglement': use_disentanglement,
            'selected_groups': [int(g) if isinstance(g, (int, np.integer)) else g for g in selected_groups] if selected_groups is not None else None,
            'selected_groups_names': selected_names if selected_names is not None else None,
            'selection_method': args.selection_method if use_disentanglement else None,
            'discretization_method': args.discretization_method if (use_disentanglement and args.selection_method == 'darts') else None,
            'darts_final_weights': darts_weights if darts_weights is not None else None,
            'average_method': args.average_method,
            'feature_dim': feature_dim,
            'num_classes_total': num_classes,
            'num_classes_evaluated': eval_num_classes,
            'best_val_auc': best_val_auc,
            'test_accuracy': acc,
            'test_auc': auc_val,
            'sensitivity': sens,
            'specificity': spec,
            'precision': prec,
            'f1_score': f1,
            'class_incremental': args.class_incremental,
            'drop_class': args.drop_class if args.class_incremental else None,
            'phase': phase_name,
        }
    else:
        config_save = {
            'mode': args.mode,
            'dataset': args.dataset,
            'backbone': args.backbone,
            'pretrained': args.pretrained,
            'freeze_backbone': args.freeze_backbone,
            'num_classes': num_classes,
            'best_val_auc': best_val_auc,
            'test_accuracy': acc,
            'test_auc': auc_val,
            'sensitivity': sens,
            'specificity': spec,
            'precision': prec,
            'f1_score': f1,
            'average_method': args.average_method,
            'class_incremental': args.class_incremental,
            'drop_class': args.drop_class if args.class_incremental else None,
            'phase': phase_name,
        }

    config_save = to_python(config_save)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config_save, f, indent=2)

    logger.info("\n" + "="*70)
    logger.info("FILES SAVED")
    logger.info("="*70)
    logger.info(f"Output directory: {output_dir}")
    logger.info("  - confusion_matrix.png")
    logger.info("  - main_predict.txt")
    logger.info("  - results.csv")
    logger.info("  - roc.png")
    logger.info("  - config.json")
    logger.info("="*70)

    return {'acc': acc, 'auc': auc_val, 'sens': sens, 'spec': spec, 'prec': prec, 'f1': f1}


def run_contrastive_evaluation(encoder_model, base_train_eval, base_val_eval, base_test_eval,
                               augmentation, args, config, num_classes, device,
                               dataset_info, output_dir, is_rgb, spatial_dims, task,
                               phase_name=None):
    """Full evaluation pipeline for contrastive/disentangled encoder.

    Extract features → (DARTS group selection) → train classifier → evaluate → save results.

    Args:
        encoder_model: Frozen encoder model
        base_train_eval, base_val_eval, base_test_eval: Base MedMNIST datasets for evaluation
        output_dir: Directory to save results
        phase_name: Optional phase identifier (e.g., 'phase1', 'phase2')

    Returns:
        dict with evaluation metrics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_disentanglement = encoder_model.hparams.get('use_disentanglement', False)
    num_aug_groups = encoder_model.hparams.get('num_aug_groups', 6)
    group_size = encoder_model.hparams.get('group_size', 128)
    projection_dim = num_aug_groups * group_size if use_disentanglement else encoder_model.hparams.get('projection_dim', 512)

    # Create feature extraction datasets (clean images, no augmentation)
    class _FeatExtractDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset, transform):
            self.base_dataset = base_dataset
            self.transform = transform
        def __len__(self):
            return len(self.base_dataset)
        def __getitem__(self, idx):
            img, label = self.base_dataset[idx]
            if isinstance(img, Image.Image):
                img = np.array(img)
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=-1)
            elif len(img.shape) == 3 and img.shape[2] == 1:
                img = np.concatenate([img, img, img], axis=-1)
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = np.clip(img, 0, 255).astype(np.uint8)
            augmented = self.transform(image=img)
            return augmented['image'], label

    feature_aug = SimCLRAugmentation(
        image_resize=config['image_size'],
        image_size=config['image_size'],
        is_rgb=is_rgb,
        is_3d=(spatial_dims == 3),
        use_disentanglement=False
    )

    train_feat_dataset = _FeatExtractDataset(base_train_eval, feature_aug.base_transform)
    val_feat_dataset = _FeatExtractDataset(base_val_eval, feature_aug.base_transform)
    test_feat_dataset = _FeatExtractDataset(base_test_eval, feature_aug.base_transform)

    train_feat_loader = DataLoader(train_feat_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    val_feat_loader = DataLoader(val_feat_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    test_feat_loader = DataLoader(test_feat_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    selected_groups = None
    darts_weights = None
    selected_names = None

    # Group selection for disentangled mode
    if use_disentanglement:
        darts_classifier = None

        if args.selection_method != 'none':
            print(f"\n{'='*70}")
            print(f"[{phase_name or 'EVAL'}] USING {args.selection_method.upper()} FOR AUTOMATIC GROUP SELECTION")
            print(f"{'='*70}")

            if args.selection_method == 'darts':
                darts_result = differentiable_group_selection(
                    encoder_model, train_feat_loader, val_feat_loader, num_classes,
                    group_size, num_aug_groups, device,
                    num_epochs=args.selection_epochs, use_wandb=False,
                    average=args.average_method, output_dir=output_dir, task=task,
                    early_stopping=args.early_stopping, early_stopping_patience=args.early_stopping_patience,
                    darts_type=args.darts_type, clf_lr=args.darts_clf_lr, arch_lr=args.darts_arch_lr,
                    clf_weight_decay=args.darts_clf_weight_decay, arch_weight_decay=args.darts_arch_weight_decay,
                    clf_eta_min=args.darts_clf_eta_min, batch_size=args.darts_batch_size,
                    discretization_method=args.discretization_method,
                    classifier_epochs=args.classifier_epochs
                )

                if args.discretization_method == 'weighted':
                    selected_groups, _, darts_weights = darts_result
                else:
                    selected_groups, _, darts_weights, darts_classifier, _, _, _, _, _, _, _ = darts_result
            else:  # grid
                selected_groups, _ = grid_search_selection(
                    encoder_model, train_feat_loader, val_feat_loader, num_classes,
                    group_size, num_aug_groups, device,
                    max_epochs=args.selection_epochs, use_wandb=False,
                    average=args.average_method, output_dir=output_dir, task=task,
                    early_stopping=args.early_stopping, early_stopping_patience=args.early_stopping_patience
                )
        elif args.aug_groups:
            selected_groups = [g.strip() for g in args.aug_groups.split(',')]
        else:
            selected_groups = list(range(num_aug_groups))

    # Extract features
    print(f"\n{'='*70}")
    print(f"[{phase_name or 'EVAL'}] EXTRACTING FEATURES FOR CLASSIFIER")
    print(f"{'='*70}")

    # Handle weighted DARTS mode
    if use_disentanglement and args.selection_method == 'darts' and args.discretization_method == 'weighted':
        aug_names = ['rotation', 'translation', 'scaling', 'contrast', 'noise']
        num_specific_groups = num_aug_groups - 1

        first_train_feats, train_labels = extract_features(
            encoder_model, train_feat_loader, device,
            aug_groups=[0], group_size=group_size, use_disentanglement=True, num_aug_groups=num_aug_groups
        )
        first_val_feats, val_labels = extract_features(
            encoder_model, val_feat_loader, device,
            aug_groups=[0], group_size=group_size, use_disentanglement=True, num_aug_groups=num_aug_groups
        )
        first_test_feats, test_labels = extract_features(
            encoder_model, test_feat_loader, device,
            aug_groups=[0], group_size=group_size, use_disentanglement=True, num_aug_groups=num_aug_groups
        )

        train_shared = first_train_feats[:, group_size:]
        val_shared = first_val_feats[:, group_size:]
        test_shared = first_test_feats[:, group_size:]

        train_weighted_sum = torch.zeros(first_train_feats.shape[0], group_size)
        val_weighted_sum = torch.zeros(first_val_feats.shape[0], group_size)
        test_weighted_sum = torch.zeros(first_test_feats.shape[0], group_size)

        for group_idx in range(num_specific_groups):
            if group_idx == 0:
                train_feats, val_feats, test_feats = first_train_feats, first_val_feats, first_test_feats
            else:
                train_feats, _ = extract_features(encoder_model, train_feat_loader, device, aug_groups=[group_idx], group_size=group_size, use_disentanglement=True, num_aug_groups=num_aug_groups)
                val_feats, _ = extract_features(encoder_model, val_feat_loader, device, aug_groups=[group_idx], group_size=group_size, use_disentanglement=True, num_aug_groups=num_aug_groups)
                test_feats, _ = extract_features(encoder_model, test_feat_loader, device, aug_groups=[group_idx], group_size=group_size, use_disentanglement=True, num_aug_groups=num_aug_groups)

            weight = darts_weights[aug_names[group_idx]]
            train_weighted_sum += weight * train_feats[:, :group_size]
            val_weighted_sum += weight * val_feats[:, :group_size]
            test_weighted_sum += weight * test_feats[:, :group_size]

        train_features = torch.cat([train_shared, train_weighted_sum], dim=1)
        val_features = torch.cat([val_shared, val_weighted_sum], dim=1)
        test_features = torch.cat([test_shared, test_weighted_sum], dim=1)
        selected_names = aug_names

        classifier, best_val_auc = train_classifier(
            train_features, train_labels, val_features, val_labels,
            num_classes, device, num_epochs=args.classifier_epochs, lr=0.001,
            use_wandb=False, average=args.average_method, output_dir=output_dir, task=task,
            early_stopping=args.early_stopping, early_stopping_patience=args.early_stopping_patience
        )

        test_dataset_tensor = TensorDataset(test_features, test_labels)
        test_loader_tensor = DataLoader(test_dataset_tensor, batch_size=128, shuffle=False)
        y_true, y_pred, y_prob = classifier.get_predictions(test_loader_tensor, device)

    elif use_disentanglement and args.selection_method == 'darts' and args.discretization_method == 'topk' and darts_classifier is not None:
        # Topk mode: reuse DARTS classifier
        aug_names = ['rotation', 'translation', 'scaling', 'contrast', 'noise']
        classifier = darts_classifier

        if len(selected_groups) == 0:
            test_feats_temp, test_labels = extract_features(encoder_model, test_feat_loader, device, aug_groups=[0], group_size=group_size, use_disentanglement=True, num_aug_groups=num_aug_groups)
            test_features = test_feats_temp[:, group_size:]
            selected_names = ['shared_only']
        else:
            test_feats_temp, test_labels = extract_features(encoder_model, test_feat_loader, device, aug_groups=[0], group_size=group_size, use_disentanglement=True, num_aug_groups=num_aug_groups)
            test_shared = test_feats_temp[:, group_size:]
            test_aug_features = []
            for group_idx in selected_groups:
                test_feats_group, _ = extract_features(encoder_model, test_feat_loader, device, aug_groups=[group_idx], group_size=group_size, use_disentanglement=True, num_aug_groups=num_aug_groups, include_shared=False)
                test_aug_features.append(test_feats_group)
            test_features = torch.cat([test_shared] + test_aug_features, dim=1)
            selected_names = [aug_names[i] for i in selected_groups]

        # Extract train features for feature_dim tracking
        train_feats_temp, train_labels = extract_features(encoder_model, train_feat_loader, device, aug_groups=[0], group_size=group_size, use_disentanglement=True, num_aug_groups=num_aug_groups)
        train_features = train_feats_temp  # for feature_dim

        best_val_auc = 0.0
        test_dataset_tensor = TensorDataset(test_features, test_labels)
        test_loader_tensor = DataLoader(test_dataset_tensor, batch_size=128, shuffle=False)
        y_true, y_pred, y_prob = classifier.get_predictions(test_loader_tensor, device)

    else:
        # Standard: grid search, manual, or contrastive (no disentanglement)
        train_features, train_labels = extract_features(
            encoder_model, train_feat_loader, device,
            aug_groups=selected_groups, group_size=group_size, use_disentanglement=use_disentanglement, num_aug_groups=num_aug_groups
        )
        val_features, val_labels = extract_features(
            encoder_model, val_feat_loader, device,
            aug_groups=selected_groups, group_size=group_size, use_disentanglement=use_disentanglement, num_aug_groups=num_aug_groups
        )
        test_features, test_labels = extract_features(
            encoder_model, test_feat_loader, device,
            aug_groups=selected_groups, group_size=group_size, use_disentanglement=use_disentanglement, num_aug_groups=num_aug_groups
        )

        if use_disentanglement and selected_groups is not None:
            aug_names = ['rotation', 'translation', 'scaling', 'contrast', 'noise']
            selected_names = [aug_names[i] for i in selected_groups]

        classifier, best_val_auc = train_classifier(
            train_features, train_labels, val_features, val_labels,
            num_classes, device, num_epochs=args.classifier_epochs, lr=0.001,
            use_wandb=False, average=args.average_method, output_dir=output_dir, task=task,
            early_stopping=args.early_stopping, early_stopping_patience=args.early_stopping_patience
        )

        test_dataset_tensor = TensorDataset(test_features, test_labels)
        test_loader_tensor = DataLoader(test_dataset_tensor, batch_size=128, shuffle=False)
        y_true, y_pred, y_prob = classifier.get_predictions(test_loader_tensor, device)

    # Evaluate and save
    results = evaluate_and_save(
        y_true, y_pred, y_prob, num_classes, dataset_info, args,
        output_dir, best_val_auc=best_val_auc, darts_weights=darts_weights,
        use_disentanglement=use_disentanglement, selected_groups=selected_groups,
        selected_names=selected_names, feature_dim=train_features.shape[1],
        phase_name=phase_name
    )

    return results


class ClassFilteredDataset:
    """Wraps a MedMNIST dataset, filtering out all samples of a specified class."""
    def __init__(self, base_dataset, drop_class):
        self.base_dataset = base_dataset
        self.valid_indices = []
        for i in range(len(base_dataset)):
            _, label = base_dataset[i]
            label_val = label.item() if hasattr(label, 'item') else int(label.squeeze())
            if label_val != drop_class:
                self.valid_indices.append(i)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        return self.base_dataset[self.valid_indices[idx]]


class ClassOnlyDataset:
    """Wraps a MedMNIST dataset, keeping ONLY samples of a specified class."""
    def __init__(self, base_dataset, keep_class):
        self.base_dataset = base_dataset
        self.valid_indices = []
        for i in range(len(base_dataset)):
            _, label = base_dataset[i]
            label_val = label.item() if hasattr(label, 'item') else int(label.squeeze())
            if label_val == keep_class:
                self.valid_indices.append(i)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        return self.base_dataset[self.valid_indices[idx]]


def main():
    parser = argparse.ArgumentParser(description='Unified Training Script for MedMNIST')
    
    # Common arguments
    parser.add_argument('--mode', type=str, required=True, choices=['baseline', 'contrastive', 'disentangled'],
                       help='Training mode')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--backbone', type=str, required=True, help='Backbone architecture')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    parser.add_argument('--use_2d_for_3d', action='store_true', help='Use 2D backbone with slice aggregation')
    parser.add_argument('--slice_aggregation', type=str, default='transformer', help='Slice aggregation method')
    parser.add_argument('--project_name', type=str, default=None, help='Project name (auto-generated if None)')
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name')
    parser.add_argument('--output_dir', type=str, default='./run', help='Output directory')
    parser.add_argument('--use_wandb', action='store_true', help='Use W&B logging')
    parser.add_argument('--average_method', type=str, default='macro', choices=['micro', 'macro'],
                       help='Averaging method for multi-class metrics')

    # Dataset parameters
    parser.add_argument('--medmnist_size', type=int, default=None, choices=[28, 64, 128, 224],
                       help='Original image size from MedMNIST+ (28, 64, 128, 224). '
                            'If not specified, uses default size (28). '
                            'Also sets default values for --image_resize and --image_size if they are not specified.')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--image_size', type=int, default=None,
                       help='Image size after crop in augmentation pipeline. '
                            'If not specified, defaults to --medmnist_size (or 28 if --medmnist_size also not specified)')
    parser.add_argument('--image_resize', type=int, default=None,
                       help='Image size for initial resize in augmentation pipeline. '
                            'If not specified, defaults to --medmnist_size (or 28 if --medmnist_size also not specified)')
    
    # Epoch controls
    parser.add_argument('--num_epochs', type=int, default=None, help='Number of epochs for main training')
    parser.add_argument('--classifier_epochs', type=int, default=100,
                       help='[Contrastive/Disentangled] Epochs to train final classifier')
    parser.add_argument('--selection_epochs', type=int, default=30,
                       help='[Disentangled] Epochs for group selection phase')
    
    # Baseline-specific
    parser.add_argument('--freeze_backbone', action='store_true', help='[Baseline] Freeze backbone weights')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512, 256],
                       help='[Baseline] Hidden dimensions for MLP classifier')
    
    # Contrastive learning arguments
    parser.add_argument('--contrastive_loss', type=str, default='simclr',
                       choices=['simclr', 'barlow_twins', 'vicreg', 'byol', 'moco'],
                       help='[Contrastive/Disentangled] Contrastive learning method')
    parser.add_argument('--projection_dim', type=int, default=None,
                       help='Projection dimension (auto-set if None)')
    parser.add_argument('--temperature', type=float, default=0.5,
                       help='[Contrastive/Disentangled] Temperature for NT-Xent loss (SimCLR/MoCo)')
    parser.add_argument('--num_aug_groups', type=int, default=6,
                       help='[Disentangled] Number of augmentation groups')

    # Method-specific hyperparameters
    parser.add_argument('--barlow_lambda', type=float, default=0.005,
                       help='[Barlow Twins] Weight for off-diagonal terms (default: 0.005)')
    parser.add_argument('--vicreg_sim_weight', type=float, default=25.0,
                       help='[VICReg] Invariance loss weight (default: 25.0)')
    parser.add_argument('--vicreg_var_weight', type=float, default=25.0,
                       help='[VICReg] Variance loss weight (default: 25.0)')
    parser.add_argument('--vicreg_cov_weight', type=float, default=1.0,
                       help='[VICReg] Covariance loss weight (default: 1.0)')
    parser.add_argument('--byol_momentum', type=float, default=0.996,
                       help='[BYOL] Momentum for target network (default: 0.996)')
    parser.add_argument('--moco_momentum', type=float, default=0.999,
                       help='[MoCo] Momentum for key encoder (default: 0.999)')
    parser.add_argument('--moco_queue_size', type=int, default=65536,
                       help='[MoCo] Queue size for negative samples (default: 65536)')
    parser.add_argument('--moco_temperature', type=float, default=0.07,
                       help='[MoCo] Temperature for contrastive loss (default: 0.07)')
    
    # True Disentanglement Loss Weights
    parser.add_argument('--inactive_weight', type=float, default=1.0,
                       help='[Disentangled] Weight for inactive identity loss (default: 1.0)')
    parser.add_argument('--ortho_weight', type=float, default=0.1,
                       help='[Disentangled] Weight for orthogonality loss between groups (default: 0.1)')
    parser.add_argument('--uniform_weight', type=float, default=0.05,
                       help='[Disentangled] Weight for uniformity loss on inactive groups (default: 0.05)')
    parser.add_argument('--uniform_t', type=float, default=2.0,
                       help='[Disentangled] Temperature for uniformity loss (default: 2.0)')
    parser.add_argument('--group_size', type=int, default=128,
                       help='[Disentangled] Dimension of each group projection (default: 128)')
    
    # NEW: Warmup schedule parameters
    parser.add_argument('--warmup_epochs', type=int, default=0,
                       help='[Disentangled] Number of warmup epochs (0 = no warmup)')
    parser.add_argument('--warmup_start_factor', type=float, default=0.0,
                       help='[Disentangled] Starting factor for warmup')

    # NEW: Encoder checkpoint selection metric
    parser.add_argument('--encoder_selection_metric', type=str, default='nt_xent',
                       choices=['nt_xent', 'total'],
                       help='[Disentangled] Metric for selecting best encoder checkpoint: '
                            '"nt_xent" uses val_nt_xent_loss (semantic features), '
                            '"total" uses val_loss (includes disentanglement penalties)')


    # Classifier training arguments
    parser.add_argument('--selection_method', type=str, default='darts',
                       choices=['darts', 'grid', 'none'],
                       help='[Disentangled] Method for automatic group selection')
    parser.add_argument('--discretization_method', type=str, default='topk',
                       choices=['topk', 'weighted'],
                       help='[Disentangled] DARTS training and discretization method: '
                            '"topk" uses CONCATENATION during DARTS (768 dims), then selects top-k groups; '
                            '"weighted" uses WEIGHTED SUM during DARTS (256 dims), applies sqrt(weight) scaling')
    parser.add_argument('--aug_groups', type=str, default=None,
                       help='[Disentangled] Comma-separated list of augmentation groups')

    # NEW: DARTS-specific hyperparameters
    parser.add_argument('--darts_type', type=str, default='first_order',
                       choices=['first_order', 'second_order'],
                       help='[DARTS] Type of DARTS optimization: '
                            '"first_order" (faster, less memory), '
                            '"second_order" (more accurate, 2x memory)')
    parser.add_argument('--darts_clf_lr', type=float, default=0.001,
                       help='[DARTS] Classifier learning rate (default: 0.001)')
    parser.add_argument('--darts_arch_lr', type=float, default=0.01,
                       help='[DARTS] Architecture learning rate (default: 0.01)')
    parser.add_argument('--darts_clf_weight_decay', type=float, default=1e-4,
                       help='[DARTS] Classifier weight decay (default: 1e-4)')
    parser.add_argument('--darts_arch_weight_decay', type=float, default=0.0,
                       help='[DARTS] Architecture weight decay (default: 0.0)')
    parser.add_argument('--darts_clf_eta_min', type=float, default=1e-5,
                       help='[DARTS] Minimum LR for classifier scheduler (default: 1e-5)')
    parser.add_argument('--darts_batch_size', type=int, default=128,
                       help='[DARTS] Batch size for DARTS training (default: 128)')

    # Early stopping
    parser.add_argument('--early_stopping', action='store_true',
                       help='Enable early stopping for all training (default: False)')
    parser.add_argument('--early_stopping_patience', type=int, default=50,
                       help='Early stopping patience (default: 50)')

    # Class incremental evaluation
    parser.add_argument('--class_incremental', action='store_true',
                       help='Enable class incremental evaluation: drop one class from encoder/baseline training, '
                            'train classifier on full dataset')
    parser.add_argument('--drop_class', type=int, default=None,
                       help='[Class Incremental] Class index to drop from encoder/baseline training')
    parser.add_argument('--phase2_epochs', type=int, default=50,
                       help='[Class Incremental] Epochs for phase 2 training on dropped class (default: 50)')

    args = parser.parse_args()

    # Validate class incremental args
    if args.class_incremental and args.drop_class is None:
        parser.error("--drop_class is required when --class_incremental is enabled")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine actual MedMNIST size to use
    actual_medmnist_size = args.medmnist_size if args.medmnist_size is not None else 28

    # Set image_resize and image_size defaults based on medmnist_size
    if args.image_resize is None:
        args.image_resize = actual_medmnist_size
    if args.image_size is None:
        args.image_size = actual_medmnist_size

    # Load dataset info (with specified size)
    _, dataset_info = get_medmnist_dataset(args.dataset, split='train', download=True, size=args.medmnist_size)

    spatial_dims = dataset_info.get('spatial_dims', 2)
    is_rgb = dataset_info.get('is_rgb', True)
    num_classes = len(dataset_info['label'])
    actual_size = dataset_info.get('actual_size', 28)

    # Validate drop_class range
    if args.class_incremental and (args.drop_class < 0 or args.drop_class >= num_classes):
        parser.error(f"--drop_class must be in range [0, {num_classes-1}], got {args.drop_class}")

    print(f"\n{'='*70}")
    print(f"MEDMNIST+ CONFIGURATION")
    print(f"{'='*70}")
    print(f"Original image size loaded: {actual_size}x{actual_size}")
    print(f"Augmentation resize size: {args.image_resize}x{args.image_resize}")
    print(f"Augmentation crop size: {args.image_size}x{args.image_size}")
    print(f"{'='*70}\n")

    # Set default hyperparameters
    if args.batch_size is None:
        args.batch_size = 4 if (spatial_dims == 3 and not args.use_2d_for_3d) else 128

    if args.learning_rate is None:
        args.learning_rate = 1e-6 if args.pretrained else 1e-4

    if args.num_epochs is None:
        args.num_epochs = 500
    
    if args.projection_dim is None:
        if args.mode == 'baseline':
            args.projection_dim = 512
        elif args.mode == 'disentangled':
            args.projection_dim = 768
        else:
            args.projection_dim = 512
    
    # Auto-detect use_2d_for_3d
    use_2d_for_3d = args.use_2d_for_3d
    if not use_2d_for_3d and spatial_dims == 3 and args.backbone.startswith('dinov2'):
        use_2d_for_3d = True
        print(f"Auto-detected: Using 2D {args.backbone} with slice aggregation")
    
    # Generate project name automatically
    if args.project_name is None:
        if args.mode == 'disentangled':
            # New architecture: true disentanglement with gradient control
            proj_dim = args.num_aug_groups * args.group_size
            args.project_name = f"medmnist_{args.mode}_true_g{args.group_size}_iw{args.inactive_weight}_ow{args.ortho_weight}_uw{args.uniform_weight}_{args.discretization_method}"
            if args.warmup_epochs > 0:
                args.project_name += f"_wu{args.warmup_epochs}"
        elif args.mode == 'contrastive':
            # Include projection dimension for contrastive mode
            args.project_name = f"medmnist_{args.mode}_ph{args.projection_dim}"
        else:
            # Baseline mode - also include projection dimension since it uses MLP
            args.project_name = f"medmnist_{args.mode}_ph{args.projection_dim}"
    
    # Append class incremental suffix to project name
    if args.class_incremental:
        args.project_name += f"_ci{args.drop_class}"

    print(f"\n{'='*70}")
    print(f"PROJECT NAME: {args.project_name}")
    print(f"{'='*70}\n")

    # Auto-generate experiment name
    if args.experiment_name is None:
        pretrain_str = "pretrained" if args.pretrained else "scratch"
        mode_str = args.mode
        if args.mode == 'baseline' and args.freeze_backbone:
            mode_str += "-frozen"

        # Add contrastive loss method to experiment name for contrastive/disentangled modes
        if args.mode in ['contrastive', 'disentangled']:
            mode_str = f"{mode_str}-{args.contrastive_loss}"

        if spatial_dims == 3 and use_2d_for_3d:
            experiment_name = f"{args.dataset}-{args.backbone}-{pretrain_str}-{mode_str}-slices-{args.slice_aggregation}"
        else:
            experiment_name = f"{args.dataset}-{args.backbone}-{pretrain_str}-{mode_str}"
    else:
        experiment_name = args.experiment_name
    
    # Create directories
    checkpoint_dir = Path(args.output_dir) / args.project_name / experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    clf_results_dir = checkpoint_dir / 'clf_results'
    clf_results_dir.mkdir(exist_ok=True)
    
    print(f"{'='*70}")
    print(f"TRAINING MODE: {args.mode.upper()}")
    print(f"{'='*70}")
    print(f"Dataset: {args.dataset} ({num_classes} classes)")
    print(f"Backbone: {args.backbone} ({'pretrained' if args.pretrained else 'from scratch'})")
    if args.mode == 'baseline':
        print(f"Backbone frozen: {args.freeze_backbone}")
    if args.mode in ['contrastive', 'disentangled']:
        print(f"Contrastive method: {args.contrastive_loss.upper()}")
    if args.mode == 'disentangled':
        print(f"Mode: TRUE DISENTANGLEMENT with Gradient Control")
        print(f"Architecture: {args.num_aug_groups - 1} aug groups + 1 semantic group × {args.group_size} dims")
        print(f"Loss Functions:")
        print(f"  - L_active: {args.contrastive_loss.upper()} loss on active group")
        print(f"  - L_semantic: {args.contrastive_loss.upper()} loss on semantic group")
        print(f"  - L_inactive: Identity loss (weight={args.inactive_weight})")
        print(f"  - L_ortho: Orthogonality (weight={args.ortho_weight})")
        print(f"  - L_uniform: Uniformity (weight={args.uniform_weight}, t={args.uniform_t})")
        if args.warmup_epochs > 0:
            print(f"Warmup: {args.warmup_epochs} epochs (start factor: {args.warmup_start_factor})")
    if args.class_incremental:
        print(f"Class Incremental: YES (dropping class {args.drop_class})")
    print(f"Output directory: {checkpoint_dir}")

    # Hyperparameters
    config = {
        'mode': args.mode,
        'dataset': args.dataset,
        'backbone': args.backbone,
        'pretrained': args.pretrained,
        'freeze_backbone': args.freeze_backbone if args.mode == 'baseline' else False,
        'spatial_dims': spatial_dims,
        'use_2d_for_3d': use_2d_for_3d,
        'slice_aggregation': args.slice_aggregation if (spatial_dims == 3 and use_2d_for_3d) else None,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'image_size': args.image_size,
        'image_resize': args.image_resize,
        'learning_rate': args.learning_rate,
        'num_classes': num_classes,
        'average_method': args.average_method,
        'projection_dim': args.projection_dim,
        'temperature': args.temperature if args.mode in ['contrastive', 'disentangled'] else None,
        'use_disentanglement': args.mode == 'disentangled',
        'num_aug_groups': args.num_aug_groups if args.mode == 'disentangled' else None,
        'group_size': args.group_size if args.mode == 'disentangled' else None,
        'inactive_weight': args.inactive_weight if args.mode == 'disentangled' else None,
        'ortho_weight': args.ortho_weight if args.mode == 'disentangled' else None,
        'uniform_weight': args.uniform_weight if args.mode == 'disentangled' else None,
        'uniform_t': args.uniform_t if args.mode == 'disentangled' else None,
        'hidden_dims': args.hidden_dims if args.mode == 'baseline' else None,
        'class_incremental': args.class_incremental,
        'drop_class': args.drop_class if args.class_incremental else None,
    }
    
    # Initialize W&B
    if args.use_wandb:
        wandb_logger = WandbLogger(
            project=args.project_name,
            name=experiment_name,
            config=config,
            log_model=False if args.mode != 'baseline' else True
        )
    else:
        wandb_logger = None
    
    # Create augmentation
    augmentation = SimCLRAugmentation(
        image_resize=config['image_size'],
        image_size=config['image_size'],
        is_rgb=is_rgb,
        is_3d=(spatial_dims == 3),
        use_disentanglement=(args.mode == 'disentangled')
    )
    
    # Load datasets
    base_train, info = get_medmnist_dataset(args.dataset, split='train', download=True, size=args.medmnist_size)
    base_val, _ = get_medmnist_dataset(args.dataset, split='val', download=True, size=args.medmnist_size)
    base_test, _ = get_medmnist_dataset(args.dataset, split='test', download=True, size=args.medmnist_size)
    
    print(f"Train samples: {len(base_train)}")
    print(f"Validation samples: {len(base_val)}")
    print(f"Test samples: {len(base_test)}")

    # Class incremental: create filtered datasets for encoder/baseline training
    if args.class_incremental:
        base_train_encoder = ClassFilteredDataset(base_train, args.drop_class)
        base_val_encoder = ClassFilteredDataset(base_val, args.drop_class)
        print(f"\nClass Incremental Mode: dropped class {args.drop_class}")
        print(f"  Encoder train samples: {len(base_train_encoder)} (was {len(base_train)})")
        print(f"  Encoder val samples: {len(base_val_encoder)} (was {len(base_val)})")
        print(f"  Classifier will use full dataset ({num_classes} classes)")
    else:
        base_train_encoder = base_train
        base_val_encoder = base_val

    # Initialize variables that will be used across all modes
    selected_groups = None
    darts_weights = None  # Will store final DARTS weights if using DARTS

    if args.mode == 'baseline':
        # BASELINE MODE
        class SupervisedDataset(torch.utils.data.Dataset):
            """Dataset WITH augmentation for training."""
            def __init__(self, base_dataset, transform):
                self.base_dataset = base_dataset
                self.transform = transform

            def __len__(self):
                return len(self.base_dataset)

            def __getitem__(self, idx):
                img, label = self.base_dataset[idx]
                view1, _ = self.transform(img)
                return view1, label

        class EvalDataset(torch.utils.data.Dataset):
            """Dataset WITHOUT augmentation for validation/testing."""
            def __init__(self, base_dataset, transform):
                self.base_dataset = base_dataset
                self.transform = transform  # Should be base_transform (no augmentation)

            def __len__(self):
                return len(self.base_dataset)

            def __getitem__(self, idx):
                img, label = self.base_dataset[idx]
                # Convert PIL Image to numpy array
                if isinstance(img, Image.Image):
                    img = np.array(img)
                # Handle grayscale images
                if len(img.shape) == 2:
                    img = np.stack([img, img, img], axis=-1)
                elif len(img.shape) == 3 and img.shape[2] == 1:
                    img = np.concatenate([img, img, img], axis=-1)
                # Ensure uint8 format
                if img.dtype != np.uint8:
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = np.clip(img, 0, 255).astype(np.uint8)
                # Apply base transform (resize, crop, normalize - NO augmentation)
                augmented = self.transform(image=img)
                return augmented['image'], label

        train_dataset = SupervisedDataset(base_train_encoder, augmentation)
        val_dataset = EvalDataset(base_val_encoder, augmentation.base_transform)  # No augmentation
        test_dataset = EvalDataset(base_test, augmentation.base_transform)  # No augmentation (always full test set)
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
        
        # Determine task type from dataset info
        task_type = info.get('task', 'multi-class')
        if 'multi-label' in task_type:
            task = 'multi-label'
        else:
            task = 'multi-class'

        model = BaselineModel(
            backbone=args.backbone,
            pretrained=args.pretrained,
            freeze_backbone=args.freeze_backbone,
            num_classes=num_classes,
            learning_rate=config['learning_rate'],
            weight_decay=1e-4,
            max_epochs=config['num_epochs'],
            spatial_dims=spatial_dims,
            use_2d_for_3d=use_2d_for_3d,
            slice_aggregation=args.slice_aggregation,
            average=args.average_method,
            hidden_dims=args.hidden_dims,
            projection_dim=args.projection_dim,
            task=task
        )
        
        callbacks = [
            ModelCheckpoint(
                dirpath=checkpoint_dir, filename='best-{epoch:02d}-{val_loss:.4f}',
                monitor='val_auc', mode='max', save_top_k=1, save_last=True, verbose=True
            ),
            LearningRateMonitor(logging_interval='epoch')
        ]

        if args.early_stopping:
            callbacks.append(EarlyStopping(monitor='val_auc', patience=args.early_stopping_patience, mode='max', verbose=False))
        
        trainer = pl.Trainer(
            max_epochs=config['num_epochs'],
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            precision="16-mixed",
            logger=wandb_logger,
            callbacks=callbacks,
            log_every_n_steps=1,
            val_check_interval=1.0,
            enable_progress_bar=True
        )
        
        print(f"\n{'='*70}")
        print(f"TRAINING BASELINE MODEL")
        print(f"{'='*70}\n")
        
        trainer.fit(model, train_loader, val_loader)
        
        best_ckpt = [c for c in callbacks if isinstance(c, ModelCheckpoint) and 'best-' in str(c.best_model_path)]
        if best_ckpt and best_ckpt[0].best_model_path:
            model = BaselineModel.load_from_checkpoint(best_ckpt[0].best_model_path)
            best_val_auc = best_ckpt[0].best_model_score.item()
        else:
            best_val_auc = trainer.callback_metrics.get('val_auc', 0.0)
            if isinstance(best_val_auc, torch.Tensor):
                best_val_auc = best_val_auc.item()
        
        print(f"Best validation AUC: {best_val_auc:.4f}")

    else:
        # CONTRASTIVE / DISENTANGLED MODE
        # Determine task type from dataset info
        task_type = info.get('task', 'multi-class')
        if 'multi-label' in task_type:
            task = 'multi-label'
        else:
            task = 'multi-class'

        train_dataset = ContrastiveDataset(base_train_encoder, augmentation, use_disentanglement=(args.mode == 'disentangled'))
        val_dataset = ContrastiveDataset(base_val_encoder, augmentation, use_disentanglement=(args.mode == 'disentangled'))
        
        if args.mode == 'disentangled':
            collate_fn = create_disentangled_collate_fn(augmentation)
            print("✓ Using custom collate function for uniform-per-batch augmentation")
        else:
            collate_fn = None
        
        train_loader = DataLoader(
            train_dataset, batch_size=config['batch_size'], shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True, collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=config['batch_size'], shuffle=False,
            num_workers=4, pin_memory=True, drop_last=True, collate_fn=collate_fn
        )
        
        print(f"✓ Train batches: {len(train_loader)}, Val batches: {len(val_loader)}\n")

        # Select contrastive learning method
        method_mapping = {
            'simclr': SimCLR,
            'barlow_twins': BarlowTwins,
            'vicreg': VICReg,
            'byol': BYOL,
            'moco': MoCo
        }

        model_class = method_mapping[args.contrastive_loss]

        # Common parameters for all methods
        common_params = {
            'backbone': config['backbone'],
            'pretrained': config['pretrained'],
            'projection_dim': config['projection_dim'],
            'learning_rate': config['learning_rate'],
            'weight_decay': 1e-4,
            'max_epochs': config['num_epochs'],
            'spatial_dims': config['spatial_dims'],
            'use_2d_for_3d': config['use_2d_for_3d'],
            'slice_aggregation': config['slice_aggregation'] if config['slice_aggregation'] else 'mean',
            # Disentanglement parameters
            'use_disentanglement': config['use_disentanglement'],
            'num_aug_groups': config['num_aug_groups'] if config['num_aug_groups'] else 6,
            'group_size': args.group_size,
            # Loss weights
            'inactive_weight': args.inactive_weight,
            'ortho_weight': args.ortho_weight,
            'uniform_weight': args.uniform_weight,
            'uniform_t': args.uniform_t,
            # Warmup
            'warmup_epochs': args.warmup_epochs,
            'warmup_start_factor': args.warmup_start_factor,
        }

        # Method-specific parameters
        method_specific_params = {}

        if args.contrastive_loss == 'simclr':
            method_specific_params = {
                'temperature': config['temperature']
            }
        elif args.contrastive_loss == 'barlow_twins':
            method_specific_params = {
                'lambd': args.barlow_lambda
            }
        elif args.contrastive_loss == 'vicreg':
            method_specific_params = {
                'sim_loss_weight': args.vicreg_sim_weight,
                'var_loss_weight': args.vicreg_var_weight,
                'cov_loss_weight': args.vicreg_cov_weight,
                'variance_threshold': 1.0
            }
        elif args.contrastive_loss == 'byol':
            method_specific_params = {
                'momentum': args.byol_momentum
            }
        elif args.contrastive_loss == 'moco':
            method_specific_params = {
                'queue_size': args.moco_queue_size,
                'momentum': args.moco_momentum,
                'temperature': args.moco_temperature
            }

        # Initialize model with combined parameters
        encoder_model = model_class(**common_params, **method_specific_params)

        print(f"\n{'='*70}")
        print(f"SELECTED CONTRASTIVE METHOD: {args.contrastive_loss.upper()}")
        print(f"{'='*70}\n")
        
        # Determine which metric to monitor for encoder checkpoint selection
        if args.mode == 'disentangled' and args.encoder_selection_metric == 'nt_xent':
            encoder_monitor_metric = 'val/primary_loss'  # L_active + L_semantic
            metric_desc = 'Primary loss (active + semantic NT-Xent)'
        else:
            # For contrastive mode or when total is selected
            encoder_monitor_metric = 'val/loss'
            metric_desc = 'Total loss'

        print(f"✓ Encoder checkpoint selection metric: {metric_desc}")

        callbacks = [
            ModelCheckpoint(
                dirpath=checkpoint_dir, filename='best-{epoch:02d}-{val_loss:.4f}',
                monitor=encoder_monitor_metric, mode='min', save_top_k=1, save_last=True, verbose=True
            ),
            LearningRateMonitor(logging_interval='epoch'),
            TQDMProgressBar(refresh_rate=10)
        ]

        if args.early_stopping:
            callbacks.append(EarlyStopping(monitor=encoder_monitor_metric, patience=args.early_stopping_patience, mode='min', verbose=False))
        
        trainer = pl.Trainer(
            max_epochs=config['num_epochs'],
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            precision="16-mixed",
            logger=wandb_logger,
            callbacks=callbacks,
            log_every_n_steps=1,
            val_check_interval=1.0,
            enable_progress_bar=True,
            enable_model_summary=True,
        )
        
        print(f"\n{'='*70}")
        method_display = args.contrastive_loss.upper().replace('_', ' ')
        print(f"TRAINING ENCODER ({'DISENTANGLED ' if args.mode == 'disentangled' else ''}{method_display})")
        print(f"{'='*70}\n")
        
        trainer.fit(encoder_model, train_loader, val_loader)
        
        # Load best encoder
        checkpoint_files = list(checkpoint_dir.glob('best-*.ckpt'))
        if len(checkpoint_files) > 0:
            checkpoint_path = checkpoint_files[0]
        else:
            checkpoint_path = checkpoint_dir / 'last.ckpt'
        
        print(f"\nLoading best encoder from: {checkpoint_path}")
        encoder_model = model_class.load_from_checkpoint(checkpoint_path)
        encoder_model.to(device)
        encoder_model.eval()
        
        for param in encoder_model.parameters():
            param.requires_grad = False
        
        if not args.class_incremental:
            # === STANDARD MODE: Single evaluation ===
            run_contrastive_evaluation(
                encoder_model, base_train, base_val, base_test,
                augmentation, args, config, num_classes, device,
                dataset_info, clf_results_dir, is_rgb, spatial_dims, task,
                phase_name=None
            )
        else:
            # === CLASS INCREMENTAL MODE: Two-phase evaluation ===
            phase1_dir = clf_results_dir / 'phase1'
            phase2_dir = clf_results_dir / 'phase2'

            # --- Phase 1: Evaluate on seen classes (N-1 classes) ---
            print(f"\n{'='*70}")
            print(f"CLASS INCREMENTAL - PHASE 1 EVALUATION")
            print(f"Evaluating encoder on {num_classes - 1} seen classes (dropped class {args.drop_class})")
            print(f"{'='*70}")

            phase1_train = ClassFilteredDataset(base_train, args.drop_class)
            phase1_val = ClassFilteredDataset(base_val, args.drop_class)
            phase1_test = ClassFilteredDataset(base_test, args.drop_class)

            run_contrastive_evaluation(
                encoder_model, phase1_train, phase1_val, phase1_test,
                augmentation, args, config, num_classes, device,
                dataset_info, phase1_dir, is_rgb, spatial_dims, task,
                phase_name='phase1'
            )

            # Save phase 1 encoder checkpoint
            phase1_ckpt = checkpoint_dir / 'phase1_best.ckpt'
            torch.save(encoder_model.state_dict(), phase1_ckpt)
            print(f"Phase 1 encoder saved to: {phase1_ckpt}")

            # --- Phase 2: Train encoder on dropped class only ---
            print(f"\n{'='*70}")
            print(f"CLASS INCREMENTAL - PHASE 2 ENCODER TRAINING")
            print(f"Training on class {args.drop_class} only for {args.phase2_epochs} epochs (no replay)")
            print(f"{'='*70}")

            # Create phase 2 datasets (only the dropped class)
            phase2_base_train = ClassOnlyDataset(base_train, args.drop_class)
            phase2_base_val = ClassOnlyDataset(base_val, args.drop_class)

            print(f"  Phase 2 train samples: {len(phase2_base_train)}")
            print(f"  Phase 2 val samples: {len(phase2_base_val)}")

            phase2_train_dataset = ContrastiveDataset(phase2_base_train, augmentation, use_disentanglement=(args.mode == 'disentangled'))
            phase2_val_dataset = ContrastiveDataset(phase2_base_val, augmentation, use_disentanglement=(args.mode == 'disentangled'))

            if args.mode == 'disentangled':
                collate_fn_p2 = create_disentangled_collate_fn(augmentation)
            else:
                collate_fn_p2 = None

            phase2_train_loader = DataLoader(
                phase2_train_dataset, batch_size=config['batch_size'], shuffle=True,
                num_workers=4, pin_memory=True, drop_last=True, collate_fn=collate_fn_p2
            )
            phase2_val_loader = DataLoader(
                phase2_val_dataset, batch_size=config['batch_size'], shuffle=False,
                num_workers=4, pin_memory=True, drop_last=True, collate_fn=collate_fn_p2
            )

            # Create fresh model with phase2 max_epochs (for correct LR scheduler)
            phase2_common_params = dict(common_params)
            phase2_common_params['max_epochs'] = args.phase2_epochs
            phase2_encoder = model_class(**phase2_common_params, **method_specific_params)

            # Load phase 1 weights
            phase2_encoder.load_state_dict(encoder_model.state_dict(), strict=False)
            print("Loaded phase 1 encoder weights into phase 2 model")

            # Unfreeze for training
            for param in phase2_encoder.parameters():
                param.requires_grad = True

            phase2_checkpoint_dir = checkpoint_dir / 'phase2'
            phase2_checkpoint_dir.mkdir(parents=True, exist_ok=True)

            phase2_callbacks = [
                ModelCheckpoint(
                    dirpath=phase2_checkpoint_dir, filename='best-{epoch:02d}-{val_loss:.4f}',
                    monitor=encoder_monitor_metric, mode='min', save_top_k=1, save_last=True, verbose=True
                ),
                LearningRateMonitor(logging_interval='epoch'),
                TQDMProgressBar(refresh_rate=10)
            ]

            if args.early_stopping:
                phase2_callbacks.append(EarlyStopping(monitor=encoder_monitor_metric, patience=args.early_stopping_patience, mode='min', verbose=False))

            phase2_trainer = pl.Trainer(
                max_epochs=args.phase2_epochs,
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                devices=1,
                precision="16-mixed",
                logger=None,
                callbacks=phase2_callbacks,
                log_every_n_steps=1,
                val_check_interval=1.0,
                enable_progress_bar=True,
                enable_model_summary=False,
            )

            phase2_trainer.fit(phase2_encoder, phase2_train_loader, phase2_val_loader)

            # Load best phase 2 encoder
            phase2_ckpt_files = list(phase2_checkpoint_dir.glob('best-*.ckpt'))
            if len(phase2_ckpt_files) > 0:
                phase2_ckpt_path = phase2_ckpt_files[0]
            else:
                phase2_ckpt_path = phase2_checkpoint_dir / 'last.ckpt'

            print(f"\nLoading best phase 2 encoder from: {phase2_ckpt_path}")
            phase2_encoder = model_class.load_from_checkpoint(phase2_ckpt_path)
            phase2_encoder.to(device)
            phase2_encoder.eval()
            for param in phase2_encoder.parameters():
                param.requires_grad = False

            # --- Phase 2: Evaluate on ALL classes ---
            print(f"\n{'='*70}")
            print(f"CLASS INCREMENTAL - PHASE 2 EVALUATION")
            print(f"Evaluating on ALL {num_classes} classes (testing catastrophic forgetting)")
            print(f"{'='*70}")

            run_contrastive_evaluation(
                phase2_encoder, base_train, base_val, base_test,
                augmentation, args, config, num_classes, device,
                dataset_info, phase2_dir, is_rgb, spatial_dims, task,
                phase_name='phase2'
            )

    if args.mode == 'baseline' and not args.class_incremental:
        # Baseline non-CIL: evaluate normally
        y_true, y_pred, y_prob = model.get_predictions(test_loader)
        evaluate_and_save(
            y_true, y_pred, y_prob, num_classes, dataset_info, args,
            clf_results_dir, best_val_auc=best_val_auc,
            phase_name=None
        )
    elif args.mode == 'baseline' and args.class_incremental:
        # Baseline CIL: Phase 1 already trained, now evaluate + Phase 2
        phase1_dir = clf_results_dir / 'phase1'
        phase2_dir = clf_results_dir / 'phase2'

        # Phase 1: evaluate on seen classes
        print(f"\n{'='*70}")
        print(f"CLASS INCREMENTAL - PHASE 1 EVALUATION (BASELINE)")
        print(f"{'='*70}")

        phase1_test = ClassFilteredDataset(base_test, args.drop_class)
        phase1_test_dataset = EvalDataset(phase1_test, augmentation.base_transform)
        phase1_test_loader = DataLoader(phase1_test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

        y_true_p1, y_pred_p1, y_prob_p1 = model.get_predictions(phase1_test_loader)
        evaluate_and_save(
            y_true_p1, y_pred_p1, y_prob_p1, num_classes, dataset_info, args,
            phase1_dir, best_val_auc=best_val_auc,
            phase_name='phase1'
        )

        # Phase 2: continue training on dropped class only
        print(f"\n{'='*70}")
        print(f"CLASS INCREMENTAL - PHASE 2 TRAINING (BASELINE)")
        print(f"Training on class {args.drop_class} only for {args.phase2_epochs} epochs")
        print(f"{'='*70}")

        phase2_base_train = ClassOnlyDataset(base_train, args.drop_class)
        phase2_base_val = ClassOnlyDataset(base_val, args.drop_class)

        phase2_train_dataset = SupervisedDataset(phase2_base_train, augmentation)
        phase2_val_dataset = EvalDataset(phase2_base_val, augmentation.base_transform)

        phase2_train_loader = DataLoader(phase2_train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
        phase2_val_loader = DataLoader(phase2_val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

        # Reset optimizer by creating fresh trainer
        phase2_checkpoint_dir = checkpoint_dir / 'phase2'
        phase2_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        phase2_callbacks = [
            ModelCheckpoint(
                dirpath=phase2_checkpoint_dir, filename='best-{epoch:02d}-{val_loss:.4f}',
                monitor='val_auc', mode='max', save_top_k=1, save_last=True, verbose=True
            ),
            LearningRateMonitor(logging_interval='epoch')
        ]
        if args.early_stopping:
            phase2_callbacks.append(EarlyStopping(monitor='val_auc', patience=args.early_stopping_patience, mode='max', verbose=False))

        phase2_trainer = pl.Trainer(
            max_epochs=args.phase2_epochs,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            precision="16-mixed",
            logger=None,
            callbacks=phase2_callbacks,
            log_every_n_steps=1,
            val_check_interval=1.0,
            enable_progress_bar=True
        )

        phase2_trainer.fit(model, phase2_train_loader, phase2_val_loader)

        # Load best phase 2 model
        phase2_ckpt_files = list(phase2_checkpoint_dir.glob('best-*.ckpt'))
        if phase2_ckpt_files:
            model = BaselineModel.load_from_checkpoint(phase2_ckpt_files[0])

        # Phase 2: evaluate on ALL classes
        print(f"\n{'='*70}")
        print(f"CLASS INCREMENTAL - PHASE 2 EVALUATION (BASELINE)")
        print(f"{'='*70}")

        test_dataset_full = EvalDataset(base_test, augmentation.base_transform)
        test_loader_full = DataLoader(test_dataset_full, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

        y_true_p2, y_pred_p2, y_prob_p2 = model.get_predictions(test_loader_full)
        evaluate_and_save(
            y_true_p2, y_pred_p2, y_prob_p2, num_classes, dataset_info, args,
            phase2_dir, best_val_auc=best_val_auc,
            phase_name='phase2'
        )

    if args.use_wandb:
        wandb.finish()

    print("\n✅ Training complete!")


if __name__ == '__main__':
    main()