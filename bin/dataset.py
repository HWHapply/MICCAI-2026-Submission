import torch
import os
import warnings
warnings.filterwarnings("ignore", message="pkg_resources")
warnings.filterwarnings('ignore', message='xFormers is not available')
warnings.filterwarnings('ignore', message='Checkpoint directory.*exists and is not empty')
torch.set_float32_matmul_precision('high')
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
torch.backends.cuda.matmul.fp32_precision = "tf32"
torch.backends.cudnn.conv.fp32_precision = "tf32"
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from PIL import Image
import random

class SimCLRAugmentation:
    """
    Customizable augmentation pipeline for SimCLR with DISENTANGLEMENT support.
    Can apply selective augmentations for learning disentangled representations.
    """
    def __init__(
        self,
        image_resize: int = 512,
        image_size: int = 448,
        is_rgb: bool = True,
        is_3d: bool = False,
        use_disentanglement: bool = False
    ):
        """
        Args:
            image_resize: Size to resize images to
            image_size: Size to crop images to
            is_rgb: Whether the images are RGB (True) or grayscale (False)
            is_3d: If True, this is for 3D data (will be handled as slices)
            use_disentanglement: If True, apply selective augmentations
        """
        self.is_3d = is_3d
        self.is_rgb = is_rgb
        self.image_size = image_size
        self.image_resize = image_resize
        self.use_disentanglement = use_disentanglement

        # Define augmentation types for disentanglement
        # 5 specific groups + 1 shared space ('basic' is now the shared representation)
        self.aug_types = ['rotation', 'translation', 'scaling', 'contrast', 'noise']
        self.num_specific_groups = 5
        self.has_shared_space = True  # 'basic' group is the shared space
        
        # Get base transforms (without specific augmentations)
        if is_rgb:
            self.base_transform = self._get_base_rgb_transform(image_size, image_resize)
            self.aug_transforms = self._get_rgb_augmentation_dict()
        else:
            self.base_transform = self._get_base_grayscale_transform(image_size, image_resize)
            self.aug_transforms = self._get_grayscale_augmentation_dict()
        
        # Full transform for non-disentangled mode
        if is_rgb:
            self.train_transform = self._get_rgb_augmentation(image_size, image_resize)
        else:
            self.train_transform = self._get_grayscale_augmentation(image_size, image_resize)
    
    def _get_base_rgb_transform(self, image_size: int, image_resize: int):
        """Base transform without augmentations (RGB)."""
        return A.Compose([
            A.Resize(image_resize, image_resize),
            A.CenterCrop(image_size, image_size),
            A.Normalize(normalization='standard'),
            ToTensorV2(),
        ])
    
    def _get_base_grayscale_transform(self, image_size: int, image_resize: int):
        """Base transform without augmentations (Grayscale)."""
        return A.Compose([
            A.Resize(image_resize, image_resize),
            A.CenterCrop(image_size, image_size),
            A.Normalize(normalization='standard'),
            ToTensorV2(),
        ])
    
    def _get_rgb_augmentation_dict(self):
        """
        Dictionary of individual augmentation transforms for RGB.

        EXPERIMENTAL DESIGN:
        - 4 groups are SAFE (balanced strength): rotation, translation, scaling, contrast
        - 1 group is HARMFUL: noise (intentionally destructive)

        All safe augmentations are tuned to similar strength so DARTS weights are balanced.
        """
        return {
            # SAFE GROUP 1: Rotation (mild)
            'rotation': A.Compose([
                A.Affine(
                    rotate=(-15, 15),  # Mild rotation
                    scale=1.0,
                    translate_percent=0.0,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=1.0
                ),
            ]),

            # SAFE GROUP 2: Translation (mild positional shift)
            'translation': A.Compose([
                A.Affine(
                    rotate=0,
                    scale=1.0,
                    translate_percent=(-0.15, 0.15),  # 15% shift
                    border_mode=cv2.BORDER_CONSTANT,
                    p=1.0
                ),
            ]),

            # SAFE GROUP 3: Scaling (mild)
            'scaling': A.Compose([
                A.Affine(
                    rotate=0,
                    scale=(0.9, 1.1),  # ±10% scaling
                    translate_percent=0.0,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=1.0
                ),
            ]),

            # SAFE GROUP 4: Contrast (stronger color/brightness changes)
            'contrast': A.Compose([
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,  # Increased from 0.1
                    contrast_limit=0.3,    # Increased from 0.1
                    p=1.0
                ),
                A.RandomGamma(
                    gamma_limit=(80, 120),  # Slightly stronger gamma
                    p=1.0
                )
            ]),

            # Shared space (no augmentation)
            'basic': A.Compose([
                # No augmentation, just resize and crop (handled by base transform)
            ]),

            # Noise augmentation group (reasonable strength)
            'noise': A.Compose([
                A.GaussNoise(
                    std_range=(0.3, 0.5),   # std from 0.3 to 0.5
                    mean_range=(0.0, 0.0),  # mean = 0 (zero-centered noise)
                    per_channel=True,
                    p=1.0
                ),
            ]),
        }

    def _get_grayscale_augmentation_dict(self):
        """
        Dictionary of individual augmentation transforms for Grayscale.

        EXPERIMENTAL DESIGN:
        - 4 groups are SAFE (balanced strength): rotation, translation, scaling, contrast
        - 1 group is HARMFUL: noise (intentionally destructive)

        All safe augmentations are tuned to similar strength so DARTS weights are balanced.
        """
        return {
            # SAFE GROUP 1: Rotation (mild)
            'rotation': A.Compose([
                A.Affine(
                    rotate=(-15, 15),  # Mild rotation
                    scale=1.0,
                    translate_percent=0.0,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=1.0
                ),
            ]),

            # SAFE GROUP 2: Translation (mild positional shift)
            'translation': A.Compose([
                A.Affine(
                    rotate=0,
                    scale=1.0,
                    translate_percent=(-0.15, 0.15),  # 15% shift
                    border_mode=cv2.BORDER_CONSTANT,
                    p=1.0
                ),
            ]),

            # SAFE GROUP 3: Scaling (mild)
            'scaling': A.Compose([
                A.Affine(
                    rotate=0,
                    scale=(0.9, 1.1),  # ±10% scaling
                    translate_percent=0.0,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=1.0
                ),
            ]),

            # SAFE GROUP 4: Contrast (CLAHE + Gamma for grayscale)
            'contrast': A.Compose([
                A.CLAHE(
                    clip_limit=(2.0, 6.0),  # Variable CLAHE strength
                    tile_grid_size=(8, 8),
                    p=1.0
                ),
                A.RandomGamma(
                    gamma_limit=(70, 130),  # Gamma correction (more natural for grayscale)
                    p=1.0
                ),
            ]),

            # Shared space (no augmentation)
            'basic': A.Compose([
                # No augmentation, just resize and crop (handled by base transform)
            ]),

            # Noise augmentation group (reasonable strength)
            'noise': A.Compose([
                A.GaussNoise(
                    std_range=(0.3, 0.5),   # std from 0.3 to 0.5
                    mean_range=(0.0, 0.0),  # mean = 0 (zero-centered noise)
                    per_channel=True,
                    p=1.0
                ),
            ]),
        }

    def _get_rgb_augmentation(self, image_size: int, image_resize: int):
        """
        RGB augmentation pipeline for non-disentangled mode (baseline/contrastive).

        WARNING: This pipeline applies ALL augmentations including HARMFUL noise,
        which will hurt performance. This is intentional for experimental validation.
        """
        return A.Compose([
            A.Resize(image_resize, image_resize),
            A.CenterCrop(image_size, image_size),

            # SAFE transforms (mild, balanced)
            A.Affine(
                rotate=(-15, 15),  # Mild rotation
                scale=(0.9, 1.1),  # Mild scaling
                translate_percent=(-0.15, 0.15),  # Mild translation
                border_mode=cv2.BORDER_CONSTANT,
                p=1.0
            ),

            A.RandomBrightnessContrast(
                brightness_limit=0.3,  # Increased from 0.1
                contrast_limit=0.3,    # Increased from 0.1
                p=1.0
            ),

            A.RandomGamma(
                gamma_limit=(80, 120),  # Slightly stronger gamma
                p=1.0
            ),

            # Noise transform
            A.GaussNoise(
                std_range=(0.3, 0.5),   # std from 0.3 to 0.5
                mean_range=(0.0, 0.0),  # mean = 0 (zero-centered noise)
                per_channel=True,
                p=1.0
            ),

            A.Normalize(normalization='standard'),
            ToTensorV2(),
        ])

    def _get_grayscale_augmentation(self, image_size: int, image_resize: int):
        """
        Grayscale augmentation pipeline for non-disentangled mode (baseline/contrastive).

        WARNING: This pipeline applies ALL augmentations including HARMFUL noise,
        which will hurt performance. This is intentional for experimental validation.
        """
        return A.Compose([
            A.Resize(image_resize, image_resize),
            A.CenterCrop(image_size, image_size),

            # SAFE transforms (mild, balanced)
            A.Affine(
                rotate=(-15, 15),  # Mild rotation
                scale=(0.9, 1.1),  # Mild scaling
                translate_percent=(-0.15, 0.15),  # Mild translation
                border_mode=cv2.BORDER_CONSTANT,
                p=1.0
            ),

            A.CLAHE(
                clip_limit=(2.0, 6.0),  # Variable CLAHE strength
                tile_grid_size=(8, 8),
                p=1.0
            ),

            A.RandomGamma(
                gamma_limit=(70, 130),  # Gamma correction (more natural for grayscale)
                p=1.0
            ),

            # Noise transform
            A.GaussNoise(
                std_range=(0.3, 0.5),   # std from 0.3 to 0.5
                mean_range=(0.0, 0.0),  # mean = 0 (zero-centered noise)
                per_channel=True,
                p=1.0
            ),

            A.Normalize(normalization='standard'),
            ToTensorV2(),
        ])

    def apply_selective_augmentation(self, image, aug_type):
        """
        Apply only ONE specific augmentation type.
        This is called by the collate function with a specific augmentation.
        
        Args:
            image: numpy array 
                - 2D: (H, W, C) or (H, W) for grayscale
                - 3D: (C, D, H, W) or (D, H, W, C) for volumes
            aug_type: one of ['rotation', 'translation', 'scaling', 'contrast', 'basic']
        
        Returns:
            augmented image tensor
        """
        # Ensure image is numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Check if this is 3D data
        if len(image.shape) == 4:
            raise NotImplementedError(
                f"3D data (shape {image.shape}) is not supported in disentangled mode. "
                "Please use standard mode (use_disentanglement=False) for 3D datasets."
            )
        
        # Handle different 2D input shapes
        if len(image.shape) == 2:
            # Grayscale (H, W) -> (H, W, 3)
            image = np.stack([image, image, image], axis=-1)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            # Grayscale (H, W, 1) -> (H, W, 3)
            image = np.concatenate([image, image, image], axis=-1)
        elif len(image.shape) == 3 and image.shape[0] == 3:
            # Might be (C, H, W) -> transpose to (H, W, C)
            image = np.transpose(image, (1, 2, 0))
        
        # Ensure we have a valid 2D shape
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Invalid image shape: {image.shape}. Expected (H, W, 3) for 2D images. "
                f"For 3D data, use standard mode (use_disentanglement=False)."
            )
        
        # Ensure uint8 format
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)
        
        # Ensure non-zero dimensions
        if image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError(f"Image has zero dimensions: {image.shape}")
        
        # First apply base transform (resize, crop)
        base_transform_no_tensor = A.Compose([
            A.Resize(self.image_resize, self.image_resize),
            A.CenterCrop(self.image_size, self.image_size),
        ])
        
        image = base_transform_no_tensor(image=image)['image']
        
        # Apply specific augmentation (skip for 'basic')
        if aug_type in self.aug_transforms and aug_type != 'basic':
            image = self.aug_transforms[aug_type](image=image)['image']
        
        # Normalize and convert to tensor
        final_transform = A.Compose([
            A.Normalize(normalization='standard'),
            ToTensorV2(),
        ])
        
        image = final_transform(image=image)['image']
        return image
    
    def apply_all_augmentations(self, image):
        """
        Apply ALL augmentations for semantic group training.
        This creates views with all augmentations applied (like standard SimCLR).

        Args:
            image: numpy array (H, W, C) or (H, W) for grayscale

        Returns:
            augmented image tensor
        """
        # Ensure image is numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Handle different 2D input shapes
        if len(image.shape) == 2:
            # Grayscale (H, W) -> (H, W, 3)
            image = np.stack([image, image, image], axis=-1)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            # Grayscale (H, W, 1) -> (H, W, 3)
            image = np.concatenate([image, image, image], axis=-1)
        elif len(image.shape) == 3 and image.shape[0] == 3:
            # Might be (C, H, W) -> transpose to (H, W, C)
            image = np.transpose(image, (1, 2, 0))

        # Ensure uint8 format
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)

        # Use the full train_transform which applies ALL augmentations
        return self.train_transform(image=image)['image']

    def _apply_to_slice(self, slice_2d):
        """Apply 2D augmentation to a single slice."""
        if slice_2d.dtype != np.uint8:
            if slice_2d.max() <= 1.0:
                slice_2d = (slice_2d * 255).astype(np.uint8)
            else:
                slice_2d = slice_2d.astype(np.uint8)

        return self.train_transform(image=slice_2d)['image']
    
    def __call__(self, x):
        """
        Apply augmentation to create two views.
        For disentangled mode, this is NOT used - we use collate_fn instead.
        
        Args:
            x: PIL Image or numpy array
        
        Returns:
            For non-disentangled: (view1, view2)
        """
        # Convert to numpy if PIL Image
        if isinstance(x, Image.Image):
            x = np.array(x)
        elif isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        
        # Handle 3D data (D, H, W, C) - apply augmentation slice-by-slice
        if len(x.shape) == 4:
            x = np.transpose(x, (1, 2, 3, 0))
            depth = x.shape[0]
            
            view1_slices = []
            for d in range(depth):
                slice_2d = x[d]
                aug_slice = self._apply_to_slice(slice_2d)
                view1_slices.append(aug_slice)
            view1 = torch.stack(view1_slices, dim=1)
            
            view2_slices = []
            for d in range(depth):
                slice_2d = x[d]
                aug_slice = self._apply_to_slice(slice_2d)
                view2_slices.append(aug_slice)
            view2 = torch.stack(view2_slices, dim=1)
            
            return view1, view2
        
        # 2D images - standard mode only (disentangled uses collate_fn)
        view1 = self.train_transform(image=x)['image']
        view2 = self.train_transform(image=x)['image']
        
        return view1, view2


class SliceAggregator(nn.Module):
    """
    Aggregates 2D slice features for 3D volumes.
    Takes a 2D backbone and processes 3D volumes slice-by-slice.
    """
    def __init__(self, backbone_2d: nn.Module, aggregation_method: str = 'mean'):
        """
        Args:
            backbone_2d: 2D backbone model (e.g., ResNet, DinoV2)
            aggregation_method: How to aggregate slice features 
                               ('mean', 'max', 'attention', 'transformer')
        """
        super().__init__()
        self.backbone_2d = backbone_2d
        self.aggregation_method = aggregation_method
        
        if aggregation_method == 'attention':
            self.attention = None
        elif aggregation_method == 'transformer':
            self.transformer = None
            self.cls_token = None
    
    def _init_attention(self, feature_dim):
        """Initialize simple attention module."""
        if self.attention is None:
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 4),
                nn.ReLU(),
                nn.Linear(feature_dim // 4, 1)
            )
    
    def _init_transformer(self, feature_dim, num_layers=2, num_heads=8, dropout=0.1):
        """Initialize transformer encoder for inter-slice attention."""
        if self.transformer is None:
            self.positional_encoding = nn.Parameter(
                torch.randn(1, 100, feature_dim) * 0.02
            )
            
            self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=num_heads,
                dim_feedforward=feature_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers
            )
            
            self.norm = nn.LayerNorm(feature_dim)
    
    def forward(self, x):
        """
        Args:
            x: Input volume of shape (batch_size, channels, depth, height, width)
        
        Returns:
            aggregated_features: (batch_size, feature_dim)
        """
        batch_size, channels, depth, height, width = x.shape
        
        x_slices = x.permute(0, 2, 1, 3, 4).contiguous()
        x_slices = x_slices.view(batch_size * depth, channels, height, width)
        
        slice_features = self.backbone_2d(x_slices)
        
        feature_dim = slice_features.shape[1]
        
        slice_features = slice_features.view(batch_size, depth, feature_dim)
        
        if self.aggregation_method == 'mean':
            aggregated = slice_features.mean(dim=1)
        
        elif self.aggregation_method == 'max':
            aggregated = slice_features.max(dim=1)[0]
        
        elif self.aggregation_method == 'attention':
            self._init_attention(feature_dim)
            self.attention = self.attention.to(slice_features.device)
            
            attention_scores = self.attention(slice_features)
            attention_weights = F.softmax(attention_scores, dim=1)
            
            aggregated = (slice_features * attention_weights).sum(dim=1)
        
        elif self.aggregation_method == 'transformer':
            self._init_transformer(feature_dim)
            self.positional_encoding = self.positional_encoding.to(slice_features.device)
            self.cls_token = self.cls_token.to(slice_features.device)
            self.transformer = self.transformer.to(slice_features.device)
            self.norm = self.norm.to(slice_features.device)
            
            slice_features = slice_features + self.positional_encoding[:, :depth, :]
            
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            slice_features = torch.cat([cls_tokens, slice_features], dim=1)
            
            transformer_output = self.transformer(slice_features)
            
            aggregated = self.norm(transformer_output[:, 0, :])
        
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
        
        return aggregated


class ContrastiveDataset(torch.utils.data.Dataset):
    """
    Wrapper dataset for contrastive learning.
    In disentangled mode, just returns raw images - augmentation happens in collate_fn.
    In standard mode, applies augmentation here.
    """
    def __init__(self, base_dataset, transform, use_disentanglement=False):
        self.base_dataset = base_dataset
        self.transform = transform
        self.use_disentanglement = use_disentanglement
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        
        if self.use_disentanglement:
            # Return raw image - augmentation will be applied in collate_fn
            # Convert to numpy for consistency
            if isinstance(img, Image.Image):
                img = np.array(img)
            elif isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            return img, label
        else:
            # Standard mode: apply augmentation here
            view1, view2 = self.transform(img)
            return (view1, view2), label


def create_disentangled_collate_fn(transform):
    """
    Create a custom collate function for uniform-per-batch augmentation.
    All samples in a batch will use the SAME augmentation type for active group.
    Additionally generates ALL-augmentation views for semantic group.

    Args:
        transform: SimCLRAugmentation instance

    Returns:
        collate_fn: Function to use in DataLoader
    """
    def collate_fn(batch):
        """
        Custom collate function that applies:
        1. Same single augmentation to all samples (for active group)
        2. ALL augmentations to all samples (for semantic group)

        Args:
            batch: List of (image, label) tuples from dataset

        Returns:
            ((view1_active, view2_active, view1_semantic, view2_semantic), labels, active_aug)
            - view1_active, view2_active: single augmentation views (for active group)
            - view1_semantic, view2_semantic: all augmentation views (for semantic group)
            - active_aug: SINGLE STRING (same for entire batch)
        """
        # Separate images and labels
        images = [item[0] for item in batch]
        labels = [item[1] for item in batch]

        # Convert labels to tensor
        labels = torch.tensor(np.array(labels))

        # Randomly choose ONE augmentation for the entire batch (5 specific groups)
        # Note: 'basic' is not included here - it's the shared space, not a selectable augmentation
        aug_types = ['rotation', 'translation', 'scaling', 'contrast', 'noise']
        batch_aug_type = random.choice(aug_types)

        # Apply augmentations to all images in batch
        view1_active_list = []
        view2_active_list = []
        view1_semantic_list = []
        view2_semantic_list = []

        for idx, img in enumerate(images):
            try:
                # Ensure image is numpy array
                if isinstance(img, Image.Image):
                    img = np.array(img)

                # Debug: check image properties
                if img.size == 0:
                    raise ValueError(f"Image {idx} is empty")

                # Apply SINGLE augmentation twice for active group
                view1_active = transform.apply_selective_augmentation(img.copy(), batch_aug_type)
                view2_active = transform.apply_selective_augmentation(img.copy(), batch_aug_type)

                # Apply ALL augmentations twice for semantic group
                view1_semantic = transform.apply_all_augmentations(img.copy())
                view2_semantic = transform.apply_all_augmentations(img.copy())

                view1_active_list.append(view1_active)
                view2_active_list.append(view2_active)
                view1_semantic_list.append(view1_semantic)
                view2_semantic_list.append(view2_semantic)
            except Exception as e:
                print(f"Error processing image {idx}:")
                print(f"  Image type: {type(img)}")
                print(f"  Image shape: {img.shape if hasattr(img, 'shape') else 'N/A'}")
                print(f"  Image dtype: {img.dtype if hasattr(img, 'dtype') else 'N/A'}")
                print(f"  Image min/max: {img.min()}/{img.max() if hasattr(img, 'min') else 'N/A'}")
                print(f"  Augmentation: {batch_aug_type}")
                raise e

        # Stack into batches
        view1_active_batch = torch.stack(view1_active_list)
        view2_active_batch = torch.stack(view2_active_list)
        view1_semantic_batch = torch.stack(view1_semantic_list)
        view2_semantic_batch = torch.stack(view2_semantic_list)

        # Return format: ((view1_active, view2_active, view1_semantic, view2_semantic), labels, active_aug)
        return (view1_active_batch, view2_active_batch, view1_semantic_batch, view2_semantic_batch), labels, batch_aug_type

    return collate_fn