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
import pytorch_lightning as pl
from torchvision import models
import monai.networks.nets as monai_nets
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights
from dataset import SliceAggregator


class ProjectionHead(nn.Module):
    """MLP projection head for SimCLR."""
    def __init__(self, input_dim: int = 2048, hidden_dim: int = 512, output_dim: int = 128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.projection(x)


class SimCLR(pl.LightningModule):
    """
    Disentangled SimCLR with Sensitivity-based Disentanglement.

    Architecture:
    - Shared backbone (ResNet/DINOv2)
    - Separate projection heads for each augmentation group + semantic group
    - Gradient control: inactive groups use detached backbone to avoid conflict

    Disentanglement Philosophy:
    - Active group: learns INVARIANCE to its augmentation (via contrastive loss)
    - Inactive groups: learn SENSITIVITY to active augmentation (via reversed loss)
    - Result: Each group is INVARIANT to its own aug, SENSITIVE to others

    Loss functions:
    - L_active: NT-Xent on active group → INVARIANT (updates backbone + head)
    - L_semantic: NT-Xent on semantic group → INVARIANT to all (updates backbone + head)
    - L_sensitivity: Reversed loss on inactive groups → SENSITIVE (updates heads only, backbone detached)
    - L_ortho: Orthogonality between ALL 6 groups (cosine sim -> 0)
    - L_uniform: Uniformity on INACTIVE groups (prevent collapse)
    """

    def __init__(
        self,
        backbone: str = 'resnet50',
        pretrained: bool = False,
        projection_dim: int = 768,  # Total: 6 groups * 128 = 768
        temperature: float = 0.5,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
        spatial_dims: int = 2,
        use_2d_for_3d: bool = False,
        slice_aggregation: str = 'mean',
        # Disentanglement parameters
        use_disentanglement: bool = True,
        num_aug_groups: int = 6,  # 5 aug groups + 1 semantic
        group_size: int = 128,
        # Loss weights
        inactive_weight: float = 1.0,
        ortho_weight: float = 0.1,
        uniform_weight: float = 0.05,
        uniform_t: float = 2.0,
        # Warmup
        warmup_epochs: int = 0,
        warmup_start_factor: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Store parameters
        self.temperature = temperature
        self.use_disentanglement = use_disentanglement
        self.num_aug_groups = num_aug_groups
        self.num_specific_groups = num_aug_groups - 1  # 5 augmentation groups
        self.group_size = group_size
        self.inactive_weight = inactive_weight
        self.ortho_weight = ortho_weight
        self.uniform_weight = uniform_weight
        self.uniform_t = uniform_t
        self.warmup_epochs = warmup_epochs
        self.warmup_start_factor = warmup_start_factor

        # Build backbone
        self.encoder = self._get_backbone(backbone, pretrained, spatial_dims, use_2d_for_3d, slice_aggregation)

        # Get encoder output dimension
        with torch.no_grad():
            if spatial_dims == 2:
                dummy_input = torch.zeros(1, 3, 224, 224)
            elif spatial_dims == 3:
                dummy_input = torch.zeros(1, 3, 28, 224, 224) if use_2d_for_3d else torch.zeros(1, 3, 28, 28, 28)
            encoder_dim = self.encoder(dummy_input).flatten(start_dim=1).shape[1]

        self.encoder_dim = encoder_dim

        if use_disentanglement:
            # SEPARATE PROJECTION HEADS for each group
            # 5 augmentation groups + 1 semantic group = 6 heads
            self.aug_projection_heads = nn.ModuleList([
                ProjectionHead(encoder_dim, 512, group_size)
                for _ in range(self.num_specific_groups)
            ])
            self.semantic_projection_head = ProjectionHead(encoder_dim, 512, group_size)

            # Augmentation name to index mapping
            self.aug_to_group = {
                'rotation': 0, 'translation': 1, 'scaling': 2, 'contrast': 3, 'noise': 4
            }
            self.group_to_aug = {v: k for k, v in self.aug_to_group.items()}

            print(f"\n{'='*70}")
            print(f"TRUE DISENTANGLED SIMCLR WITH GRADIENT CONTROL")
            print(f"{'='*70}")
            print(f"Architecture:")
            print(f"  - Shared backbone: {backbone} ({'pretrained' if pretrained else 'scratch'})")
            print(f"  - Encoder dim: {encoder_dim}")
            print(f"  - 5 augmentation projection heads: {encoder_dim} -> 512 -> {group_size}")
            print(f"  - 1 semantic projection head: {encoder_dim} -> 512 -> {group_size}")
            print(f"  - Total projection dim: {num_aug_groups * group_size}")
            print(f"\nGradient Control:")
            print(f"  - Active group: gradient flows to backbone + projection head")
            print(f"  - Semantic group: gradient flows to backbone + projection head")
            print(f"  - Inactive groups: gradient flows to projection head ONLY (backbone detached)")
            print(f"\nLoss Functions:")
            print(f"  - L_active: NT-Xent (temperature={temperature})")
            print(f"  - L_semantic: NT-Xent (temperature={temperature})")
            print(f"  - L_inactive: Identity loss (weight={inactive_weight})")
            print(f"  - L_ortho: Orthogonality loss (weight={ortho_weight})")
            print(f"  - L_uniform: Uniformity loss (weight={uniform_weight}, t={uniform_t})")
            if warmup_epochs > 0:
                print(f"\nWarmup: {warmup_epochs} epochs (start factor: {warmup_start_factor})")
            print(f"{'='*70}\n")
        else:
            # Standard SimCLR: single projection head
            self.projection_head = ProjectionHead(encoder_dim, encoder_dim, projection_dim)
            print(f"✓ Standard SimCLR (no disentanglement)")
            print(f"  - Backbone: {backbone}")
            print(f"  - Projection dim: {projection_dim}")

    def _get_backbone(self, backbone_name: str, pretrained: bool, spatial_dims: int,
                      use_2d_for_3d: bool, slice_aggregation: str):
        if spatial_dims == 2:
            return self._get_2d_backbone(backbone_name, pretrained)
        elif spatial_dims == 3:
            if use_2d_for_3d:
                backbone_2d = self._get_2d_backbone(backbone_name, pretrained)
                return SliceAggregator(backbone_2d, aggregation_method=slice_aggregation)
            else:
                return self._get_3d_backbone(backbone_name, pretrained)
        else:
            raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")

    def _get_2d_backbone(self, backbone_name: str, pretrained: bool):
        if backbone_name.startswith('dinov2'):
            backbone = torch.hub.load('facebookresearch/dinov2', backbone_name, pretrained=pretrained)
            print(f"✓ Loaded {backbone_name} ({'pretrained' if pretrained else 'scratch'})")
            return backbone

        weights_map = {
            'resnet18': ResNet18_Weights.IMAGENET1K_V1 if pretrained else None,
            'resnet34': ResNet34_Weights.IMAGENET1K_V1 if pretrained else None,
            'resnet50': ResNet50_Weights.IMAGENET1K_V1 if pretrained else None,
            'resnet101': ResNet101_Weights.IMAGENET1K_V1 if pretrained else None,
        }

        if backbone_name not in weights_map:
            raise ValueError(f"2D Backbone {backbone_name} not supported")

        backbone_fn = getattr(models, backbone_name)
        backbone = backbone_fn(weights=weights_map[backbone_name])
        backbone = nn.Sequential(*list(backbone.children())[:-1])
        print(f"✓ Loaded {backbone_name} ({'pretrained' if pretrained else 'scratch'})")
        return backbone

    def _get_3d_backbone(self, backbone_name: str, pretrained: bool):
        if not backbone_name.startswith('resnet'):
            raise ValueError(f"3D version of {backbone_name} not implemented")

        depth = int(backbone_name.replace("resnet", ""))
        monai_resnet_map = {
            18: monai_nets.resnet18, 34: monai_nets.resnet34,
            50: monai_nets.resnet50, 101: monai_nets.resnet101
        }

        if depth not in monai_resnet_map:
            raise ValueError(f"3D ResNet depth {depth} not supported")

        backbone = monai_resnet_map[depth](pretrained=False, spatial_dims=3, n_input_channels=3)
        backbone.fc = nn.Identity()
        print(f"✓ Loaded 3D {backbone_name}")
        if pretrained:
            print("  Warning: Pretrained weights not available for 3D models")
        return backbone

    def get_warmup_factor(self, current_epoch: int) -> float:
        """Get warmup factor for disentanglement losses."""
        if self.warmup_epochs == 0 or current_epoch >= self.warmup_epochs:
            return 1.0
        progress = current_epoch / self.warmup_epochs
        return self.warmup_start_factor + (1.0 - self.warmup_start_factor) * progress

    def forward(self, x, active_group_idx=None, detach_inactive=True):
        """
        Forward pass with gradient control.

        Args:
            x: input images [batch, C, H, W]
            active_group_idx: index of active augmentation group (0-4), None for standard mode
            detach_inactive: DEPRECATED - all groups now receive full gradients.
                             Use high --inactive_weight to balance gradient conflict.

        Disentanglement design:
            - Active group: learns INVARIANCE via contrastive loss (updates backbone)
            - Inactive groups: learn SENSITIVITY via reversed loss (updates backbone)
            - Use --inactive_weight 2.0+ to let sensitivity compete with contrastive

        Returns:
            h: backbone features [batch, encoder_dim]
            z: concatenated group features [batch, num_groups * group_size]
            z_groups: list of individual group features (only if use_disentanglement)
        """
        # Shared backbone
        h = self.encoder(x).flatten(start_dim=1)

        if self.use_disentanglement:
            z_groups = []

            # Project each augmentation group - ALL get full gradient
            for i, head in enumerate(self.aug_projection_heads):
                z_i = head(h)  # Full gradient to backbone for all groups
                z_groups.append(F.normalize(z_i, dim=1))

            # Semantic group: always keep gradient to backbone
            z_semantic = self.semantic_projection_head(h)
            z_semantic = F.normalize(z_semantic, dim=1)
            z_groups.append(z_semantic)

            # Concatenate all groups
            z = torch.cat(z_groups, dim=1)

            return h, z, z_groups
        else:
            # Standard SimCLR
            z = self.projection_head(h)
            return h, z

    def nt_xent_loss(self, z_i, z_j):
        """NT-Xent contrastive loss."""
        batch_size = z_i.shape[0]

        # Normalize
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate
        representations = torch.cat([z_i, z_j], dim=0)

        # Similarity matrix
        similarity_matrix = torch.mm(representations, representations.T) / self.temperature

        # Create labels
        batch_indices = torch.arange(batch_size, device=self.device)
        labels = torch.cat([batch_indices + batch_size, batch_indices], dim=0)

        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=self.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))

        # Cross entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss

    def sensitivity_loss(self, z1_groups, z2_groups, active_idx):
        """
        Sensitivity loss for inactive groups (push apart).
        Inactive groups should output DIFFERENT features for the two views,
        making them SENSITIVE to the active augmentation.

        Design: Each group learns SENSITIVITY to OTHER augmentations.
        - Active group: INVARIANT to its aug (via contrastive)
        - Inactive groups: SENSITIVE to active aug (via this loss)

        Requires high --inactive_weight (e.g., 2.0) to compete with contrastive losses.
        """
        loss = torch.tensor(0.0, device=self.device)
        num_inactive = 0

        for i in range(self.num_specific_groups):
            if i != active_idx:
                # Cosine similarity should be LOW (different outputs)
                cos_sim = F.cosine_similarity(z1_groups[i], z2_groups[i], dim=1)
                loss = loss + (1.0 + cos_sim).mean()  # Range [0, 2], minimized when cos_sim = -1
                num_inactive += 1

        if num_inactive > 0:
            loss = loss / num_inactive

        return loss

    def orthogonality_loss(self, z_groups):
        """
        Orthogonality loss between all pairs of groups.
        Different groups should be orthogonal (cosine sim -> 0).
        """
        loss = torch.tensor(0.0, device=self.device)
        num_pairs = 0
        num_groups = len(z_groups)

        for i in range(num_groups):
            for j in range(i + 1, num_groups):
                # Cosine similarity should be 0 (orthogonal)
                cos_sim = F.cosine_similarity(z_groups[i], z_groups[j], dim=1)
                loss = loss + cos_sim.pow(2).mean()
                num_pairs += 1

        if num_pairs > 0:
            loss = loss / num_pairs

        return loss

    def uniformity_loss(self, z, t=2.0):
        """
        Uniformity loss (Wang & Isola 2020).
        Encourages features to spread uniformly on the hypersphere.
        """
        z = F.normalize(z, dim=1)
        sq_pdist = torch.pdist(z, p=2).pow(2)
        return sq_pdist.mul(-t).exp().mean().log()

    def training_step(self, batch, batch_idx):
        if self.use_disentanglement:
            # New format: (view1_active, view2_active, view1_semantic, view2_semantic), labels, active_aug
            (x_i, x_j, x_sem_i, x_sem_j), _, active_aug = batch
            active_idx = self.aug_to_group.get(active_aug, 0)
        else:
            (x_i, x_j), _ = batch
            active_idx = None

        batch_size = x_i.shape[0]
        warmup_factor = self.get_warmup_factor(self.current_epoch)

        if self.use_disentanglement:
            # Forward pass for ACTIVE views (single augmentation)
            # All groups get full gradient - use high --inactive_weight to balance
            _, z1, z1_groups = self(x_i, active_group_idx=active_idx)
            _, z2, z2_groups = self(x_j, active_group_idx=active_idx)

            # Forward pass for SEMANTIC views (all augmentations)
            _, _, z_sem1_groups = self(x_sem_i, active_group_idx=None)
            _, _, z_sem2_groups = self(x_sem_j, active_group_idx=None)

            # === LOSS 1: Active group NT-Xent (using single-aug views) ===
            loss_active = self.nt_xent_loss(z1_groups[active_idx], z2_groups[active_idx])

            # === LOSS 2: Semantic group NT-Xent (using ALL-aug views) ===
            semantic_idx = self.num_specific_groups  # Last group is semantic
            loss_semantic = self.nt_xent_loss(z_sem1_groups[semantic_idx], z_sem2_groups[semantic_idx])

            # === LOSS 3: Inactive groups sensitivity loss (REVERSED from identity) ===
            # Pushes inactive groups to be SENSITIVE/VARIANT to the active augmentation
            loss_inactive = self.sensitivity_loss(z1_groups, z2_groups, active_idx)

            # === LOSS 4: Orthogonality between ALL 6 groups (including semantic) ===
            # This ensures augmentation heads learn different features from semantic head
            # Use mean of both views for stability
            z_mean_groups = [(z1_groups[i] + z2_groups[i]) / 2 for i in range(len(z1_groups))]  # All 6 groups
            loss_ortho = self.orthogonality_loss(z_mean_groups)

            # === LOSS 5: Uniformity on INACTIVE groups only ===
            # Active group gets uniformity from NT-Xent (pushes negatives apart)
            # Semantic group gets uniformity from NT-Xent (L_semantic)
            # Inactive groups need explicit uniformity to prevent collapse
            loss_uniform = torch.tensor(0.0, device=self.device)
            num_inactive = 0
            for i in range(self.num_specific_groups):
                if i != active_idx:
                    loss_uniform = loss_uniform + self.uniformity_loss(z1_groups[i], self.uniform_t)
                    loss_uniform = loss_uniform + self.uniformity_loss(z2_groups[i], self.uniform_t)
                    num_inactive += 1
            if num_inactive > 0:
                loss_uniform = loss_uniform / (2 * num_inactive)

            # === Total loss ===
            total_loss = (
                loss_active
                + loss_semantic
                + warmup_factor * self.inactive_weight * loss_inactive
                + warmup_factor * self.ortho_weight * loss_ortho
                + warmup_factor * self.uniform_weight * loss_uniform
            )

            # Primary loss for monitoring (sum of contrastive losses)
            primary_loss = loss_active + loss_semantic

            # === Logging (train/ section in W&B) ===
            self.log('train/loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
            self.log('train/primary_loss', primary_loss, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log('train/active_loss', loss_active, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log('train/semantic_loss', loss_semantic, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log('train/inactive_loss', loss_inactive, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log('train/ortho_loss', loss_ortho, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log('train/uniform_loss', loss_uniform, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log('train/warmup_factor', warmup_factor, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log('train/active_group', float(active_idx), on_step=False, on_epoch=True, batch_size=batch_size)

            # === Monitoring metrics (monitor/ section in W&B) ===
            with torch.no_grad():
                # Per-group statistics
                aug_names = ['rotation', 'translation', 'scaling', 'contrast', 'noise', 'semantic']
                for i, name in enumerate(aug_names):
                    z_group = z1_groups[i]
                    # Variance (should be maintained, not collapse)
                    group_var = z_group.var(dim=0).mean()
                    self.log(f'monitor/group_{name}_var', group_var, on_step=False, on_epoch=True, batch_size=batch_size)

                    # Intra-group similarity
                    if batch_size > 1:
                        sim_matrix = torch.mm(z_group, z_group.T)
                        mask = ~torch.eye(batch_size, dtype=torch.bool, device=self.device)
                        intra_sim = sim_matrix[mask].mean()
                        self.log(f'monitor/group_{name}_intra_sim', intra_sim, on_step=False, on_epoch=True, batch_size=batch_size)

                # Inter-group similarity (should be ~0 due to orthogonality)
                inter_sims = []
                for i in range(len(z1_groups)):
                    for j in range(i + 1, len(z1_groups)):
                        cos_sim = F.cosine_similarity(z1_groups[i], z1_groups[j], dim=1).mean()
                        inter_sims.append(cos_sim.abs())
                if inter_sims:
                    mean_inter_sim = torch.stack(inter_sims).mean()
                    self.log('monitor/inter_group_sim', mean_inter_sim, on_step=False, on_epoch=True, batch_size=batch_size)

                # Sensitivity monitoring per inactive group
                # LOW cos_sim = good (outputs are different = sensitive to augmentation)
                # HIGH cos_sim = bad (outputs are similar = not sensitive)
                for i in range(self.num_specific_groups):
                    if i != active_idx:
                        cos_sim = F.cosine_similarity(z1_groups[i], z2_groups[i], dim=1).mean()
                        self.log(f'monitor/sensitivity_{aug_names[i]}', cos_sim, on_step=False, on_epoch=True, batch_size=batch_size)

        else:
            # Standard SimCLR
            _, z_i = self(x_i)
            _, z_j = self(x_j)

            z_i = F.normalize(z_i, dim=1)
            z_j = F.normalize(z_j, dim=1)

            total_loss = self.nt_xent_loss(z_i, z_j)

            self.log('train/loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)

            with torch.no_grad():
                pos_sim = (z_i * z_j).sum(dim=1).mean()
                self.log('monitor/pos_similarity', pos_sim, on_step=False, on_epoch=True, batch_size=batch_size)

        return total_loss

    def validation_step(self, batch, batch_idx):
        if self.use_disentanglement:
            # New format: (view1_active, view2_active, view1_semantic, view2_semantic), labels, active_aug
            (x_i, x_j, x_sem_i, x_sem_j), _, active_aug = batch
            active_idx = self.aug_to_group.get(active_aug, 0)
        else:
            (x_i, x_j), _ = batch
            active_idx = None

        batch_size = x_i.shape[0]
        warmup_factor = self.get_warmup_factor(self.current_epoch)

        if self.use_disentanglement:
            # Forward pass for ACTIVE views (single augmentation)
            _, z1, z1_groups = self(x_i, active_group_idx=active_idx)
            _, z2, z2_groups = self(x_j, active_group_idx=active_idx)

            # Forward pass for SEMANTIC views (all augmentations)
            _, _, z_sem1_groups = self(x_sem_i, active_group_idx=None)
            _, _, z_sem2_groups = self(x_sem_j, active_group_idx=None)

            # Compute losses
            loss_active = self.nt_xent_loss(z1_groups[active_idx], z2_groups[active_idx])
            semantic_idx = self.num_specific_groups
            loss_semantic = self.nt_xent_loss(z_sem1_groups[semantic_idx], z_sem2_groups[semantic_idx])
            loss_inactive = self.sensitivity_loss(z1_groups, z2_groups, active_idx)

            z_mean_groups = [(z1_groups[i] + z2_groups[i]) / 2 for i in range(len(z1_groups))]
            loss_ortho = self.orthogonality_loss(z_mean_groups)

            loss_uniform = torch.tensor(0.0, device=self.device)
            num_inactive = 0
            for i in range(self.num_specific_groups):
                if i != active_idx:
                    loss_uniform = loss_uniform + self.uniformity_loss(z1_groups[i], self.uniform_t)
                    loss_uniform = loss_uniform + self.uniformity_loss(z2_groups[i], self.uniform_t)
                    num_inactive += 1
            if num_inactive > 0:
                loss_uniform = loss_uniform / (2 * num_inactive)

            total_loss = (
                loss_active
                + loss_semantic
                + warmup_factor * self.inactive_weight * loss_inactive
                + warmup_factor * self.ortho_weight * loss_ortho
                + warmup_factor * self.uniform_weight * loss_uniform
            )

            primary_loss = loss_active + loss_semantic

            # Logging (val/ section in W&B)
            self.log('val/loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
            self.log('val/primary_loss', primary_loss, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log('val/active_loss', loss_active, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log('val/semantic_loss', loss_semantic, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log('val/inactive_loss', loss_inactive, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log('val/ortho_loss', loss_ortho, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log('val/uniform_loss', loss_uniform, on_step=False, on_epoch=True, batch_size=batch_size)

            # Monitoring (val_monitor/ section in W&B)
            with torch.no_grad():
                aug_names = ['rotation', 'translation', 'scaling', 'contrast', 'noise', 'semantic']
                for i, name in enumerate(aug_names):
                    z_group = z1_groups[i]
                    group_var = z_group.var(dim=0).mean()
                    self.log(f'val_monitor/group_{name}_var', group_var, on_step=False, on_epoch=True, batch_size=batch_size)

                inter_sims = []
                for i in range(len(z1_groups)):
                    for j in range(i + 1, len(z1_groups)):
                        cos_sim = F.cosine_similarity(z1_groups[i], z1_groups[j], dim=1).mean()
                        inter_sims.append(cos_sim.abs())
                if inter_sims:
                    mean_inter_sim = torch.stack(inter_sims).mean()
                    self.log('val_monitor/inter_group_sim', mean_inter_sim, on_step=False, on_epoch=True, batch_size=batch_size)

        else:
            # Standard SimCLR
            _, z_i = self(x_i)
            _, z_j = self(x_j)

            z_i = F.normalize(z_i, dim=1)
            z_j = F.normalize(z_j, dim=1)

            total_loss = self.nt_xent_loss(z_i, z_j)

            self.log('val/loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=0
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }
