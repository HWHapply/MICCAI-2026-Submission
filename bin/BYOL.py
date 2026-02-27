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
import copy


class ProjectionHead(nn.Module):
    """MLP projection head for BYOL."""
    def __init__(self, input_dim: int = 2048, hidden_dim: int = 512, output_dim: int = 128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.projection(x)


class PredictorHead(nn.Module):
    """MLP predictor head for BYOL (on top of projection)."""
    def __init__(self, input_dim: int = 128, hidden_dim: int = 512, output_dim: int = 128):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.predictor(x)


class BYOL(pl.LightningModule):
    """
    Disentangled BYOL with Sensitivity-based Disentanglement.

    Architecture:
    - Shared backbone (ResNet/DINOv2)
    - Online network: projection heads + predictor heads
    - Target network: EMA of projection heads (no predictor)
    - Separate heads for each augmentation group + semantic group

    Disentanglement Philosophy:
    - Active group: learns INVARIANCE to its augmentation (via BYOL loss)
    - Inactive groups: learn SENSITIVITY to active augmentation (via reversed loss)
    - Result: Each group is INVARIANT to its own aug, SENSITIVE to others

    Loss functions:
    - L_active: BYOL on active group → INVARIANT (updates backbone + head + predictor)
    - L_semantic: BYOL on semantic group → INVARIANT to all (updates backbone + head + predictor)
    - L_sensitivity: Reversed loss on inactive groups → SENSITIVE (updates heads + predictors only)
    - L_ortho: Orthogonality between ALL 6 groups (cosine sim -> 0)
    - L_uniform: Uniformity on INACTIVE groups (prevent collapse)
    """

    def __init__(
        self,
        backbone: str = 'resnet50',
        pretrained: bool = False,
        projection_dim: int = 768,  # Total: 6 groups * 128 = 768
        momentum: float = 0.996,  # EMA momentum for target network
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
        self.momentum = momentum
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

        # Build backbone (online network)
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
            # ONLINE NETWORK: projection heads + predictor heads
            # 5 augmentation groups + 1 semantic group = 6 heads
            self.aug_projection_heads = nn.ModuleList([
                ProjectionHead(encoder_dim, 512, group_size)
                for _ in range(self.num_specific_groups)
            ])
            self.semantic_projection_head = ProjectionHead(encoder_dim, 512, group_size)

            # Predictor heads (one for each projection head)
            self.aug_predictor_heads = nn.ModuleList([
                PredictorHead(group_size, 512, group_size)
                for _ in range(self.num_specific_groups)
            ])
            self.semantic_predictor_head = PredictorHead(group_size, 512, group_size)

            # TARGET NETWORK: EMA of encoder + projection heads (no predictor)
            self.target_encoder = copy.deepcopy(self.encoder)
            self.target_aug_projection_heads = copy.deepcopy(self.aug_projection_heads)
            self.target_semantic_projection_head = copy.deepcopy(self.semantic_projection_head)

            # Freeze target network (updated via EMA)
            for param in self.target_encoder.parameters():
                param.requires_grad = False
            for head in self.target_aug_projection_heads:
                for param in head.parameters():
                    param.requires_grad = False
            for param in self.target_semantic_projection_head.parameters():
                param.requires_grad = False

            # Augmentation name to index mapping
            self.aug_to_group = {
                'rotation': 0, 'translation': 1, 'scaling': 2, 'contrast': 3, 'noise': 4
            }
            self.group_to_aug = {v: k for k, v in self.aug_to_group.items()}

            print(f"\n{'='*70}")
            print(f"TRUE DISENTANGLED BYOL WITH GRADIENT CONTROL")
            print(f"{'='*70}")
            print(f"Architecture:")
            print(f"  - Shared backbone: {backbone} ({'pretrained' if pretrained else 'scratch'})")
            print(f"  - Encoder dim: {encoder_dim}")
            print(f"  - Online network:")
            print(f"    - 5 augmentation projection heads: {encoder_dim} -> 512 -> {group_size}")
            print(f"    - 1 semantic projection head: {encoder_dim} -> 512 -> {group_size}")
            print(f"    - 5 augmentation predictor heads: {group_size} -> 512 -> {group_size}")
            print(f"    - 1 semantic predictor head: {group_size} -> 512 -> {group_size}")
            print(f"  - Target network: EMA of encoder + projection heads (momentum={momentum})")
            print(f"  - Total projection dim: {num_aug_groups * group_size}")
            print(f"\nGradient Control:")
            print(f"  - Active group: gradient flows to backbone + projection head + predictor")
            print(f"  - Semantic group: gradient flows to backbone + projection head + predictor")
            print(f"  - Inactive groups: gradient flows to projection head + predictor ONLY (backbone detached)")
            print(f"\nLoss Functions:")
            print(f"  - L_active: BYOL (MSE between predictor and target)")
            print(f"  - L_semantic: BYOL (MSE between predictor and target)")
            print(f"  - L_inactive: Identity loss (weight={inactive_weight})")
            print(f"  - L_ortho: Orthogonality loss (weight={ortho_weight})")
            print(f"  - L_uniform: Uniformity loss (weight={uniform_weight}, t={uniform_t})")
            if warmup_epochs > 0:
                print(f"\nWarmup: {warmup_epochs} epochs (start factor: {warmup_start_factor})")
            print(f"{'='*70}\n")
        else:
            # Standard BYOL: single projection head + predictor
            self.projection_head = ProjectionHead(encoder_dim, encoder_dim, projection_dim)
            self.predictor_head = PredictorHead(projection_dim, encoder_dim, projection_dim)

            # Target network
            self.target_encoder = copy.deepcopy(self.encoder)
            self.target_projection_head = copy.deepcopy(self.projection_head)

            # Freeze target network
            for param in self.target_encoder.parameters():
                param.requires_grad = False
            for param in self.target_projection_head.parameters():
                param.requires_grad = False

            print(f"✓ Standard BYOL (no disentanglement)")
            print(f"  - Backbone: {backbone}")
            print(f"  - Projection dim: {projection_dim}")
            print(f"  - Target momentum: {momentum}")

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

    @torch.no_grad()
    def _update_target_network(self):
        """Update target network using exponential moving average."""
        if self.use_disentanglement:
            # Update target encoder
            for online_params, target_params in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                target_params.data = self.momentum * target_params.data + (1 - self.momentum) * online_params.data

            # Update target projection heads
            for online_head, target_head in zip(self.aug_projection_heads, self.target_aug_projection_heads):
                for online_params, target_params in zip(online_head.parameters(), target_head.parameters()):
                    target_params.data = self.momentum * target_params.data + (1 - self.momentum) * online_params.data

            # Update semantic head
            for online_params, target_params in zip(
                self.semantic_projection_head.parameters(),
                self.target_semantic_projection_head.parameters()
            ):
                target_params.data = self.momentum * target_params.data + (1 - self.momentum) * online_params.data
        else:
            # Standard BYOL
            for online_params, target_params in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                target_params.data = self.momentum * target_params.data + (1 - self.momentum) * online_params.data

            for online_params, target_params in zip(
                self.projection_head.parameters(),
                self.target_projection_head.parameters()
            ):
                target_params.data = self.momentum * target_params.data + (1 - self.momentum) * online_params.data

    def forward(self, x, active_group_idx=None, detach_inactive=True, use_target=False):
        """
        Forward pass with gradient control.

        Args:
            x: input images [batch, C, H, W]
            active_group_idx: index of active augmentation group (0-4), None for standard mode
            detach_inactive: DEPRECATED - all groups now receive full gradients.
            use_target: if True, use target network instead of online network

        Returns:
            h: backbone features [batch, encoder_dim]
            z: concatenated group features [batch, num_groups * group_size]
            z_groups: list of individual group features (only if use_disentanglement)
            q_groups: list of predictor outputs (only if use_disentanglement and not use_target)
        """
        if self.use_disentanglement:
            # Select encoder and projection heads
            encoder = self.target_encoder if use_target else self.encoder
            aug_heads = self.target_aug_projection_heads if use_target else self.aug_projection_heads
            semantic_head = self.target_semantic_projection_head if use_target else self.semantic_projection_head

            # Shared backbone
            h = encoder(x).flatten(start_dim=1)

            z_groups = []

            # Project each augmentation group
            # NOTE: No normalization here - byol_loss() normalizes internally
            for i, head in enumerate(aug_heads):
                z_i = head(h)
                z_groups.append(z_i)

            # Semantic group
            z_semantic = semantic_head(h)
            z_groups.append(z_semantic)

            # Concatenate all groups
            z = torch.cat(z_groups, dim=1)

            # Apply predictor (only for online network)
            if not use_target:
                q_groups = []
                for i, predictor in enumerate(self.aug_predictor_heads):
                    q_i = predictor(z_groups[i])
                    q_groups.append(q_i)
                q_semantic = self.semantic_predictor_head(z_groups[-1])
                q_groups.append(q_semantic)

                return h, z, z_groups, q_groups
            else:
                return h, z, z_groups, None

        else:
            # Standard BYOL
            encoder = self.target_encoder if use_target else self.encoder
            projection = self.target_projection_head if use_target else self.projection_head

            h = encoder(x).flatten(start_dim=1)
            z = projection(h)

            if not use_target:
                q = self.predictor_head(z)
                return h, z, q
            else:
                return h, z, None

    def byol_loss(self, q, z_target):
        """
        BYOL loss: MSE between predictor output and target projection.

        Args:
            q: predictor output (from online network) [batch, dim]
            z_target: projection from target network (detached) [batch, dim]

        Returns:
            loss: scalar loss value
        """
        q = F.normalize(q, dim=1)
        z_target = F.normalize(z_target, dim=1)
        return 2 - 2 * (q * z_target).sum(dim=1).mean()

    def sensitivity_loss(self, q1_groups, q2_groups, active_idx):
        """
        Sensitivity loss for inactive groups (push apart).
        Inactive groups should output DIFFERENT features for the two views,
        making them SENSITIVE to the active augmentation.
        """
        loss = torch.tensor(0.0, device=self.device)
        num_inactive = 0

        for i in range(self.num_specific_groups):
            if i != active_idx:
                # Cosine similarity should be LOW (different outputs)
                cos_sim = F.cosine_similarity(q1_groups[i], q2_groups[i], dim=1)
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
            # Online network forward pass for ACTIVE views
            _, _, z1_groups, q1_groups = self(x_i, active_group_idx=active_idx, use_target=False)
            _, _, z2_groups, q2_groups = self(x_j, active_group_idx=active_idx, use_target=False)

            # Target network forward pass for ACTIVE views (detached)
            with torch.no_grad():
                _, _, z1_target_groups, _ = self(x_i, active_group_idx=active_idx, use_target=True)
                _, _, z2_target_groups, _ = self(x_j, active_group_idx=active_idx, use_target=True)

            # Online network forward pass for SEMANTIC views
            _, _, z_sem1_groups, q_sem1_groups = self(x_sem_i, active_group_idx=None, use_target=False)
            _, _, z_sem2_groups, q_sem2_groups = self(x_sem_j, active_group_idx=None, use_target=False)

            # Target network forward pass for SEMANTIC views (detached)
            with torch.no_grad():
                _, _, z_sem1_target_groups, _ = self(x_sem_i, active_group_idx=None, use_target=True)
                _, _, z_sem2_target_groups, _ = self(x_sem_j, active_group_idx=None, use_target=True)

            # === LOSS 1: Active group BYOL (using single-aug views) ===
            # Symmetric loss: q1->z2_target + q2->z1_target
            loss_active = (
                self.byol_loss(q1_groups[active_idx], z2_target_groups[active_idx])
                + self.byol_loss(q2_groups[active_idx], z1_target_groups[active_idx])
            ) / 2

            # === LOSS 2: Semantic group BYOL (using ALL-aug views) ===
            semantic_idx = self.num_specific_groups  # Last group is semantic
            loss_semantic = (
                self.byol_loss(q_sem1_groups[semantic_idx], z_sem2_target_groups[semantic_idx])
                + self.byol_loss(q_sem2_groups[semantic_idx], z_sem1_target_groups[semantic_idx])
            ) / 2

            # === LOSS 3: Inactive groups sensitivity loss ===
            loss_inactive = self.sensitivity_loss(q1_groups, q2_groups, active_idx)

            # === LOSS 4: Orthogonality between ALL 6 groups ===
            z_mean_groups = [(z1_groups[i] + z2_groups[i]) / 2 for i in range(len(z1_groups))]
            loss_ortho = self.orthogonality_loss(z_mean_groups)

            # === LOSS 5: Uniformity on INACTIVE groups only ===
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

            # Primary loss for monitoring
            primary_loss = loss_active + loss_semantic

            # === Logging ===
            self.log('train/loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
            self.log('train/primary_loss', primary_loss, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log('train/active_loss', loss_active, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log('train/semantic_loss', loss_semantic, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log('train/inactive_loss', loss_inactive, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log('train/ortho_loss', loss_ortho, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log('train/uniform_loss', loss_uniform, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log('train/warmup_factor', warmup_factor, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log('train/active_group', float(active_idx), on_step=False, on_epoch=True, batch_size=batch_size)

            # === Monitoring ===
            with torch.no_grad():
                aug_names = ['rotation', 'translation', 'scaling', 'contrast', 'noise', 'semantic']
                for i, name in enumerate(aug_names):
                    z_group = z1_groups[i]
                    group_var = z_group.var(dim=0).mean()
                    self.log(f'monitor/group_{name}_var', group_var, on_step=False, on_epoch=True, batch_size=batch_size)

                    if batch_size > 1:
                        sim_matrix = torch.mm(z_group, z_group.T)
                        mask = ~torch.eye(batch_size, dtype=torch.bool, device=self.device)
                        intra_sim = sim_matrix[mask].mean()
                        self.log(f'monitor/group_{name}_intra_sim', intra_sim, on_step=False, on_epoch=True, batch_size=batch_size)

                inter_sims = []
                for i in range(len(z1_groups)):
                    for j in range(i + 1, len(z1_groups)):
                        cos_sim = F.cosine_similarity(z1_groups[i], z1_groups[j], dim=1).mean()
                        inter_sims.append(cos_sim.abs())
                if inter_sims:
                    mean_inter_sim = torch.stack(inter_sims).mean()
                    self.log('monitor/inter_group_sim', mean_inter_sim, on_step=False, on_epoch=True, batch_size=batch_size)

                for i in range(self.num_specific_groups):
                    if i != active_idx:
                        cos_sim = F.cosine_similarity(q1_groups[i], q2_groups[i], dim=1).mean()
                        self.log(f'monitor/sensitivity_{aug_names[i]}', cos_sim, on_step=False, on_epoch=True, batch_size=batch_size)

        else:
            # Standard BYOL
            _, z_i, q_i = self(x_i, use_target=False)
            _, z_j, q_j = self(x_j, use_target=False)

            with torch.no_grad():
                _, z_i_target, _ = self(x_i, use_target=True)
                _, z_j_target, _ = self(x_j, use_target=True)

            # Symmetric loss
            total_loss = (
                self.byol_loss(q_i, z_j_target)
                + self.byol_loss(q_j, z_i_target)
            ) / 2

            self.log('train/loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)

            with torch.no_grad():
                cos_sim = F.cosine_similarity(q_i, z_j_target, dim=1).mean()
                self.log('monitor/pred_target_sim', cos_sim, on_step=False, on_epoch=True, batch_size=batch_size)

        # Update target network
        self._update_target_network()

        return total_loss

    def validation_step(self, batch, batch_idx):
        if self.use_disentanglement:
            (x_i, x_j, x_sem_i, x_sem_j), _, active_aug = batch
            active_idx = self.aug_to_group.get(active_aug, 0)
        else:
            (x_i, x_j), _ = batch
            active_idx = None

        batch_size = x_i.shape[0]
        warmup_factor = self.get_warmup_factor(self.current_epoch)

        if self.use_disentanglement:
            _, _, z1_groups, q1_groups = self(x_i, active_group_idx=active_idx, use_target=False)
            _, _, z2_groups, q2_groups = self(x_j, active_group_idx=active_idx, use_target=False)

            with torch.no_grad():
                _, _, z1_target_groups, _ = self(x_i, active_group_idx=active_idx, use_target=True)
                _, _, z2_target_groups, _ = self(x_j, active_group_idx=active_idx, use_target=True)

            _, _, z_sem1_groups, q_sem1_groups = self(x_sem_i, active_group_idx=None, use_target=False)
            _, _, z_sem2_groups, q_sem2_groups = self(x_sem_j, active_group_idx=None, use_target=False)

            with torch.no_grad():
                _, _, z_sem1_target_groups, _ = self(x_sem_i, active_group_idx=None, use_target=True)
                _, _, z_sem2_target_groups, _ = self(x_sem_j, active_group_idx=None, use_target=True)

            loss_active = (
                self.byol_loss(q1_groups[active_idx], z2_target_groups[active_idx])
                + self.byol_loss(q2_groups[active_idx], z1_target_groups[active_idx])
            ) / 2

            semantic_idx = self.num_specific_groups
            loss_semantic = (
                self.byol_loss(q_sem1_groups[semantic_idx], z_sem2_target_groups[semantic_idx])
                + self.byol_loss(q_sem2_groups[semantic_idx], z_sem1_target_groups[semantic_idx])
            ) / 2

            loss_inactive = self.sensitivity_loss(q1_groups, q2_groups, active_idx)

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

            self.log('val/loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
            self.log('val/primary_loss', primary_loss, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log('val/active_loss', loss_active, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log('val/semantic_loss', loss_semantic, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log('val/inactive_loss', loss_inactive, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log('val/ortho_loss', loss_ortho, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log('val/uniform_loss', loss_uniform, on_step=False, on_epoch=True, batch_size=batch_size)

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
            _, z_i, q_i = self(x_i, use_target=False)
            _, z_j, q_j = self(x_j, use_target=False)

            with torch.no_grad():
                _, z_i_target, _ = self(x_i, use_target=True)
                _, z_j_target, _ = self(x_j, use_target=True)

            total_loss = (
                self.byol_loss(q_i, z_j_target)
                + self.byol_loss(q_j, z_i_target)
            ) / 2

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
