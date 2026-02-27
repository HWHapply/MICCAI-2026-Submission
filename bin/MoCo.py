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
    """MLP projection head for MoCo."""
    def __init__(self, input_dim: int = 2048, hidden_dim: int = 512, output_dim: int = 128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.projection(x)


class MoCo(pl.LightningModule):
    """
    Disentangled MoCo with Sensitivity-based Disentanglement.

    Architecture:
    - Query encoder (trainable): shared backbone + projection heads
    - Key encoder (momentum update): EMA of query encoder
    - Queue: FIFO buffer of negative samples for each group
    - Separate heads for each augmentation group + semantic group

    Disentanglement Philosophy:
    - Active group: learns INVARIANCE to its augmentation (via MoCo loss)
    - Inactive groups: learn SENSITIVITY to active augmentation (via reversed loss)
    - Result: Each group is INVARIANT to its own aug, SENSITIVE to others

    Loss functions:
    - L_active: MoCo on active group → INVARIANT (updates backbone + head)
    - L_semantic: MoCo on semantic group → INVARIANT to all (updates backbone + head)
    - L_sensitivity: Reversed loss on inactive groups → SENSITIVE (updates heads only)
    - L_ortho: Orthogonality between ALL 6 groups (cosine sim -> 0)
    - L_uniform: Uniformity on INACTIVE groups (prevent collapse)
    """

    def __init__(
        self,
        backbone: str = 'resnet50',
        pretrained: bool = False,
        projection_dim: int = 768,  # Total: 6 groups * 128 = 768
        queue_size: int = 65536,  # Size of negative queue
        momentum: float = 0.999,  # Momentum for key encoder update
        temperature: float = 0.07,  # Temperature for contrastive loss
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
        self.queue_size = queue_size
        self.momentum = momentum
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

        # Build query encoder (trainable)
        self.encoder_q = self._get_backbone(backbone, pretrained, spatial_dims, use_2d_for_3d, slice_aggregation)

        # Get encoder output dimension
        with torch.no_grad():
            if spatial_dims == 2:
                dummy_input = torch.zeros(1, 3, 224, 224)
            elif spatial_dims == 3:
                dummy_input = torch.zeros(1, 3, 28, 224, 224) if use_2d_for_3d else torch.zeros(1, 3, 28, 28, 28)
            encoder_dim = self.encoder_q(dummy_input).flatten(start_dim=1).shape[1]

        self.encoder_dim = encoder_dim

        if use_disentanglement:
            # QUERY ENCODER: projection heads (trainable)
            # 5 augmentation groups + 1 semantic group = 6 heads
            self.aug_projection_heads_q = nn.ModuleList([
                ProjectionHead(encoder_dim, 512, group_size)
                for _ in range(self.num_specific_groups)
            ])
            self.semantic_projection_head_q = ProjectionHead(encoder_dim, 512, group_size)

            # KEY ENCODER: EMA of query encoder (frozen, updated via momentum)
            self.encoder_k = copy.deepcopy(self.encoder_q)
            self.aug_projection_heads_k = copy.deepcopy(self.aug_projection_heads_q)
            self.semantic_projection_head_k = copy.deepcopy(self.semantic_projection_head_q)

            # Freeze key encoder
            for param in self.encoder_k.parameters():
                param.requires_grad = False
            for head in self.aug_projection_heads_k:
                for param in head.parameters():
                    param.requires_grad = False
            for param in self.semantic_projection_head_k.parameters():
                param.requires_grad = False

            # Create queues for each group (FIFO buffers)
            # Shape: [num_groups, group_size, queue_size]
            for i in range(num_aug_groups):
                self.register_buffer(f"queue_{i}", torch.randn(group_size, queue_size))
                setattr(self, f"queue_{i}", F.normalize(getattr(self, f"queue_{i}"), dim=0))
                self.register_buffer(f"queue_ptr_{i}", torch.zeros(1, dtype=torch.long))

            # Augmentation name to index mapping
            self.aug_to_group = {
                'rotation': 0, 'translation': 1, 'scaling': 2, 'contrast': 3, 'noise': 4
            }
            self.group_to_aug = {v: k for k, v in self.aug_to_group.items()}

            print(f"\n{'='*70}")
            print(f"TRUE DISENTANGLED MOCO WITH GRADIENT CONTROL")
            print(f"{'='*70}")
            print(f"Architecture:")
            print(f"  - Shared backbone: {backbone} ({'pretrained' if pretrained else 'scratch'})")
            print(f"  - Encoder dim: {encoder_dim}")
            print(f"  - Query encoder (trainable):")
            print(f"    - 5 augmentation projection heads: {encoder_dim} -> 512 -> {group_size}")
            print(f"    - 1 semantic projection head: {encoder_dim} -> 512 -> {group_size}")
            print(f"  - Key encoder: EMA of query encoder (momentum={momentum})")
            print(f"  - Queue: {num_aug_groups} queues of size {queue_size}")
            print(f"  - Total projection dim: {num_aug_groups * group_size}")
            print(f"\nGradient Control:")
            print(f"  - Active group: gradient flows to backbone + projection head")
            print(f"  - Semantic group: gradient flows to backbone + projection head")
            print(f"  - Inactive groups: gradient flows to projection head ONLY (backbone detached)")
            print(f"\nLoss Functions:")
            print(f"  - L_active: MoCo (temperature={temperature})")
            print(f"  - L_semantic: MoCo (temperature={temperature})")
            print(f"  - L_inactive: Identity loss (weight={inactive_weight})")
            print(f"  - L_ortho: Orthogonality loss (weight={ortho_weight})")
            print(f"  - L_uniform: Uniformity loss (weight={uniform_weight}, t={uniform_t})")
            if warmup_epochs > 0:
                print(f"\nWarmup: {warmup_epochs} epochs (start factor: {warmup_start_factor})")
            print(f"{'='*70}\n")
        else:
            # Standard MoCo: single projection head
            self.projection_head_q = ProjectionHead(encoder_dim, encoder_dim, projection_dim)

            # Key encoder
            self.encoder_k = copy.deepcopy(self.encoder_q)
            self.projection_head_k = copy.deepcopy(self.projection_head_q)

            # Freeze key encoder
            for param in self.encoder_k.parameters():
                param.requires_grad = False
            for param in self.projection_head_k.parameters():
                param.requires_grad = False

            # Create queue
            self.register_buffer("queue", torch.randn(projection_dim, queue_size))
            self.queue = F.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

            print(f"✓ Standard MoCo (no disentanglement)")
            print(f"  - Backbone: {backbone}")
            print(f"  - Projection dim: {projection_dim}")
            print(f"  - Queue size: {queue_size}")
            print(f"  - Momentum: {momentum}")
            print(f"  - Temperature: {temperature}")

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
    def _momentum_update_key_encoder(self):
        """Update key encoder using exponential moving average."""
        if self.use_disentanglement:
            # Update key encoder backbone
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

            # Update key projection heads
            for head_q, head_k in zip(self.aug_projection_heads_q, self.aug_projection_heads_k):
                for param_q, param_k in zip(head_q.parameters(), head_k.parameters()):
                    param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

            # Update semantic head
            for param_q, param_k in zip(
                self.semantic_projection_head_q.parameters(),
                self.semantic_projection_head_k.parameters()
            ):
                param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)
        else:
            # Standard MoCo
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

            for param_q, param_k in zip(
                self.projection_head_q.parameters(),
                self.projection_head_k.parameters()
            ):
                param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, group_idx):
        """Update queue with new keys for a specific group."""
        batch_size = keys.shape[0]

        ptr = int(getattr(self, f"queue_ptr_{group_idx}"))
        queue = getattr(self, f"queue_{group_idx}")

        # Replace oldest batch in queue with new keys
        if ptr + batch_size <= self.queue_size:
            queue[:, ptr:ptr + batch_size] = keys.T
        else:
            # Wrap around
            remaining = self.queue_size - ptr
            queue[:, ptr:] = keys[:remaining].T
            queue[:, :batch_size - remaining] = keys[remaining:].T

        # Move pointer
        ptr = (ptr + batch_size) % self.queue_size
        getattr(self, f"queue_ptr_{group_idx}")[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_standard(self, keys):
        """Update queue for standard MoCo."""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        if ptr + batch_size <= self.queue_size:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            remaining = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, :batch_size - remaining] = keys[remaining:].T

        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def forward(self, x, active_group_idx=None, use_key_encoder=False):
        """
        Forward pass with gradient control.

        Args:
            x: input images [batch, C, H, W]
            active_group_idx: index of active augmentation group (0-4), None for standard mode
            use_key_encoder: if True, use key encoder instead of query encoder

        Returns:
            h: backbone features [batch, encoder_dim]
            z: concatenated group features [batch, num_groups * group_size]
            z_groups: list of individual group features (only if use_disentanglement)
        """
        if self.use_disentanglement:
            # Select encoder and projection heads
            encoder = self.encoder_k if use_key_encoder else self.encoder_q
            aug_heads = self.aug_projection_heads_k if use_key_encoder else self.aug_projection_heads_q
            semantic_head = self.semantic_projection_head_k if use_key_encoder else self.semantic_projection_head_q

            # Shared backbone
            h = encoder(x).flatten(start_dim=1)

            z_groups = []

            # Project each augmentation group
            for i, head in enumerate(aug_heads):
                z_i = head(h)
                z_groups.append(F.normalize(z_i, dim=1))

            # Semantic group
            z_semantic = semantic_head(h)
            z_semantic = F.normalize(z_semantic, dim=1)
            z_groups.append(z_semantic)

            # Concatenate all groups
            z = torch.cat(z_groups, dim=1)

            return h, z, z_groups
        else:
            # Standard MoCo
            encoder = self.encoder_k if use_key_encoder else self.encoder_q
            projection = self.projection_head_k if use_key_encoder else self.projection_head_q

            h = encoder(x).flatten(start_dim=1)
            z = projection(h)
            z = F.normalize(z, dim=1)

            return h, z

    def moco_loss(self, q, k, queue):
        """
        MoCo contrastive loss (InfoNCE).

        Args:
            q: query embeddings [batch, dim]
            k: key embeddings [batch, dim]
            queue: negative samples [dim, queue_size]

        Returns:
            loss: scalar loss value
        """
        # Positive logits: [batch, 1]
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

        # Negative logits: [batch, queue_size]
        l_neg = torch.einsum('nc,ck->nk', [q, queue.clone().detach()])

        # Logits: [batch, 1 + queue_size]
        logits = torch.cat([l_pos, l_neg], dim=1)

        # Apply temperature
        logits /= self.temperature

        # Labels: positives are at index 0
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=self.device)

        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)

        return loss

    def sensitivity_loss(self, z1_groups, z2_groups, active_idx):
        """
        Sensitivity loss for inactive groups (push apart).
        Inactive groups should output DIFFERENT features for the two views.
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
            # Query encoder forward pass for ACTIVE views
            _, _, q1_groups = self(x_i, active_group_idx=active_idx, use_key_encoder=False)
            _, _, q2_groups = self(x_j, active_group_idx=active_idx, use_key_encoder=False)

            # Key encoder forward pass for ACTIVE views (no gradient)
            with torch.no_grad():
                self._momentum_update_key_encoder()  # Update key encoder
                _, _, k1_groups = self(x_i, active_group_idx=active_idx, use_key_encoder=True)
                _, _, k2_groups = self(x_j, active_group_idx=active_idx, use_key_encoder=True)

            # Query encoder forward pass for SEMANTIC views
            _, _, q_sem1_groups = self(x_sem_i, active_group_idx=None, use_key_encoder=False)
            _, _, q_sem2_groups = self(x_sem_j, active_group_idx=None, use_key_encoder=False)

            # Key encoder forward pass for SEMANTIC views (no gradient)
            with torch.no_grad():
                _, _, k_sem1_groups = self(x_sem_i, active_group_idx=None, use_key_encoder=True)
                _, _, k_sem2_groups = self(x_sem_j, active_group_idx=None, use_key_encoder=True)

            # === LOSS 1: Active group MoCo (using single-aug views) ===
            queue_active = getattr(self, f"queue_{active_idx}")
            loss_active = (
                self.moco_loss(q1_groups[active_idx], k2_groups[active_idx], queue_active)
                + self.moco_loss(q2_groups[active_idx], k1_groups[active_idx], queue_active)
            ) / 2

            # === LOSS 2: Semantic group MoCo (using ALL-aug views) ===
            semantic_idx = self.num_specific_groups  # Last group is semantic
            queue_semantic = getattr(self, f"queue_{semantic_idx}")
            loss_semantic = (
                self.moco_loss(q_sem1_groups[semantic_idx], k_sem2_groups[semantic_idx], queue_semantic)
                + self.moco_loss(q_sem2_groups[semantic_idx], k_sem1_groups[semantic_idx], queue_semantic)
            ) / 2

            # === LOSS 3: Inactive groups sensitivity loss ===
            loss_inactive = self.sensitivity_loss(q1_groups, q2_groups, active_idx)

            # === LOSS 4: Orthogonality between ALL 6 groups ===
            z_mean_groups = [(q1_groups[i] + q2_groups[i]) / 2 for i in range(len(q1_groups))]
            loss_ortho = self.orthogonality_loss(z_mean_groups)

            # === LOSS 5: Uniformity on INACTIVE groups only ===
            loss_uniform = torch.tensor(0.0, device=self.device)
            num_inactive = 0
            for i in range(self.num_specific_groups):
                if i != active_idx:
                    loss_uniform = loss_uniform + self.uniformity_loss(q1_groups[i], self.uniform_t)
                    loss_uniform = loss_uniform + self.uniformity_loss(q2_groups[i], self.uniform_t)
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

            # Update queues
            with torch.no_grad():
                # Update active group queue
                keys_active = torch.cat([k1_groups[active_idx], k2_groups[active_idx]], dim=0)
                self._dequeue_and_enqueue(keys_active, active_idx)

                # Update semantic group queue
                keys_semantic = torch.cat([k_sem1_groups[semantic_idx], k_sem2_groups[semantic_idx]], dim=0)
                self._dequeue_and_enqueue(keys_semantic, semantic_idx)

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
                    z_group = q1_groups[i]
                    group_var = z_group.var(dim=0).mean()
                    self.log(f'monitor/group_{name}_var', group_var, on_step=False, on_epoch=True, batch_size=batch_size)

                    if batch_size > 1:
                        sim_matrix = torch.mm(z_group, z_group.T)
                        mask = ~torch.eye(batch_size, dtype=torch.bool, device=self.device)
                        intra_sim = sim_matrix[mask].mean()
                        self.log(f'monitor/group_{name}_intra_sim', intra_sim, on_step=False, on_epoch=True, batch_size=batch_size)

                inter_sims = []
                for i in range(len(q1_groups)):
                    for j in range(i + 1, len(q1_groups)):
                        cos_sim = F.cosine_similarity(q1_groups[i], q1_groups[j], dim=1).mean()
                        inter_sims.append(cos_sim.abs())
                if inter_sims:
                    mean_inter_sim = torch.stack(inter_sims).mean()
                    self.log('monitor/inter_group_sim', mean_inter_sim, on_step=False, on_epoch=True, batch_size=batch_size)

                for i in range(self.num_specific_groups):
                    if i != active_idx:
                        cos_sim = F.cosine_similarity(q1_groups[i], q2_groups[i], dim=1).mean()
                        self.log(f'monitor/sensitivity_{aug_names[i]}', cos_sim, on_step=False, on_epoch=True, batch_size=batch_size)

        else:
            # Standard MoCo
            _, q_i = self(x_i, use_key_encoder=False)
            _, q_j = self(x_j, use_key_encoder=False)

            with torch.no_grad():
                self._momentum_update_key_encoder()
                _, k_i = self(x_i, use_key_encoder=True)
                _, k_j = self(x_j, use_key_encoder=True)

            # Symmetric loss
            total_loss = (
                self.moco_loss(q_i, k_j, self.queue)
                + self.moco_loss(q_j, k_i, self.queue)
            ) / 2

            # Update queue
            with torch.no_grad():
                keys = torch.cat([k_i, k_j], dim=0)
                self._dequeue_and_enqueue_standard(keys)

            self.log('train/loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)

            with torch.no_grad():
                # Monitor positive similarity
                pos_sim = (q_i * k_j).sum(dim=1).mean()
                self.log('monitor/pos_similarity', pos_sim, on_step=False, on_epoch=True, batch_size=batch_size)

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
            _, _, q1_groups = self(x_i, active_group_idx=active_idx, use_key_encoder=False)
            _, _, q2_groups = self(x_j, active_group_idx=active_idx, use_key_encoder=False)

            with torch.no_grad():
                _, _, k1_groups = self(x_i, active_group_idx=active_idx, use_key_encoder=True)
                _, _, k2_groups = self(x_j, active_group_idx=active_idx, use_key_encoder=True)

            _, _, q_sem1_groups = self(x_sem_i, active_group_idx=None, use_key_encoder=False)
            _, _, q_sem2_groups = self(x_sem_j, active_group_idx=None, use_key_encoder=False)

            with torch.no_grad():
                _, _, k_sem1_groups = self(x_sem_i, active_group_idx=None, use_key_encoder=True)
                _, _, k_sem2_groups = self(x_sem_j, active_group_idx=None, use_key_encoder=True)

            queue_active = getattr(self, f"queue_{active_idx}")
            loss_active = (
                self.moco_loss(q1_groups[active_idx], k2_groups[active_idx], queue_active)
                + self.moco_loss(q2_groups[active_idx], k1_groups[active_idx], queue_active)
            ) / 2

            semantic_idx = self.num_specific_groups
            queue_semantic = getattr(self, f"queue_{semantic_idx}")
            loss_semantic = (
                self.moco_loss(q_sem1_groups[semantic_idx], k_sem2_groups[semantic_idx], queue_semantic)
                + self.moco_loss(q_sem2_groups[semantic_idx], k_sem1_groups[semantic_idx], queue_semantic)
            ) / 2

            loss_inactive = self.sensitivity_loss(q1_groups, q2_groups, active_idx)

            z_mean_groups = [(q1_groups[i] + q2_groups[i]) / 2 for i in range(len(q1_groups))]
            loss_ortho = self.orthogonality_loss(z_mean_groups)

            loss_uniform = torch.tensor(0.0, device=self.device)
            num_inactive = 0
            for i in range(self.num_specific_groups):
                if i != active_idx:
                    loss_uniform = loss_uniform + self.uniformity_loss(q1_groups[i], self.uniform_t)
                    loss_uniform = loss_uniform + self.uniformity_loss(q2_groups[i], self.uniform_t)
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
                    z_group = q1_groups[i]
                    group_var = z_group.var(dim=0).mean()
                    self.log(f'val_monitor/group_{name}_var', group_var, on_step=False, on_epoch=True, batch_size=batch_size)

                inter_sims = []
                for i in range(len(q1_groups)):
                    for j in range(i + 1, len(q1_groups)):
                        cos_sim = F.cosine_similarity(q1_groups[i], q1_groups[j], dim=1).mean()
                        inter_sims.append(cos_sim.abs())
                if inter_sims:
                    mean_inter_sim = torch.stack(inter_sims).mean()
                    self.log('val_monitor/inter_group_sim', mean_inter_sim, on_step=False, on_epoch=True, batch_size=batch_size)

        else:
            _, q_i = self(x_i, use_key_encoder=False)
            _, q_j = self(x_j, use_key_encoder=False)

            with torch.no_grad():
                _, k_i = self(x_i, use_key_encoder=True)
                _, k_j = self(x_j, use_key_encoder=True)

            total_loss = (
                self.moco_loss(q_i, k_j, self.queue)
                + self.moco_loss(q_j, k_i, self.queue)
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
