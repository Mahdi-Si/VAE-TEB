try:
    # Preferred (package) import
    from .vae_teb_model_prediction import *  # noqa: F403
except ImportError:
    # Backward-compatible fallback when running from this directory
    from vae_teb_model_prediction import *  # noqa: F403

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence, Tuple


# =====================================================================
# Focal BCE Loss with Label Smoothing
# =====================================================================


class FocalBCEWithLogitsLoss(nn.Module):
    """Focal binary cross-entropy loss with optional label smoothing.

    Combines focal loss (Lin et al. 2017) with per-bit weighting and
    label smoothing for multi-label / hierarchical classification.

    Args:
        gamma: Focusing parameter. Higher values down-weight easy
            examples more aggressively. 0 reduces to standard BCE.
        alpha: Optional per-bit weight tensor of shape ``(num_bits,)``.
        label_smoothing: Smoothing factor. Targets are moved toward 0.5
            by this amount: ``t' = t * (1 - s) + 0.5 * s``.
        reduction: ``'mean'`` | ``'sum'`` | ``'none'``.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal BCE loss.

        Args:
            logits: Raw logits of shape ``(*, num_bits)``.
            targets: Float targets of shape ``(*, num_bits)`` in ``[0, 1]``.

        Returns:
            Scalar loss (when reduction is ``'mean'`` or ``'sum'``),
            otherwise per-element loss.
        """
        if self.label_smoothing > 0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        p = torch.sigmoid(logits)
        p_t = targets * p + (1.0 - targets) * (1.0 - p)
        focal_weight = (1.0 - p_t) ** self.gamma
        loss = focal_weight * bce

        if self.alpha is not None:
            loss = loss * self.alpha

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# =====================================================================
# Attention-based temporal pooling
# =====================================================================


class SegmentAttentionPooling(nn.Module):
    """Learned attention pooling over temporal hidden states.

    Computes a weighted sum of hidden states over the time dimension,
    allowing the model to focus on clinically significant timesteps
    (e.g. decelerations, variability changes) within each segment.

    Architecture::

        score_t = w^T tanh(W h_t + b)
        alpha   = softmax(score, dim=T)
        v       = sum(alpha * h, dim=T)

    Args:
        hidden_dim: Dimension of the hidden states.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention-weighted sum of hidden states.

        Args:
            hidden_states: Tensor of shape ``(B, T, H)``.

        Returns:
            Tuple of ``(pooled, alpha)`` where ``pooled`` has shape
            ``(B, H)`` and ``alpha`` has shape ``(B, T)``.
        """
        scores = self.attention_net(hidden_states)      # (B, T, 1)
        alpha = F.softmax(scores, dim=-2)               # (B, T, 1)
        pooled = (alpha * hidden_states).sum(dim=-2)    # (B, H)
        return pooled, alpha.squeeze(-1)


# =====================================================================
# Hierarchical label utilities
# =====================================================================


def map_to_hierarchical_labels(labels: torch.Tensor) -> torch.Tensor:
    """Map scalar class labels to hierarchical multi-hot encoding.

    Encoding:
        - Healthy  (label <= 1) → ``[1, 0, 0]``
        - Acidosis (label == 2) → ``[0, 1, 0]``
        - HIE      (label == 3) → ``[0, 1, 1]``

    Args:
        labels: Integer tensor of shape ``(N,)`` with values in ``{0, 1, 2, 3}``.
            Values 0 and 1 are both treated as healthy.

    Returns:
        Float tensor of shape ``(N, 3)``.
    """
    targets = torch.zeros(labels.shape[0], 3, device=labels.device, dtype=torch.float32)
    targets[labels <= 1, 0] = 1.0   # healthy bit
    targets[labels == 2, 1] = 1.0   # unhealthy bit (acidosis)
    targets[labels == 3, 1] = 1.0   # unhealthy bit (HIE)
    targets[labels == 3, 2] = 1.0   # severe bit (HIE only)
    return targets


class BaseTimeSeriesClassifier(nn.Module):
    """Base class for time series classifiers.

    Supports two label modes controlled by ``label_mode``:
    - ``'binary'``: Standard 2-class softmax + cross-entropy.
    - ``'hierarchical'``: 3-bit multi-hot sigmoid + focal BCE.

    Args:
        input_dim: Input feature dimension per timestep.
        num_classes: Number of output units (2 for binary, 3 for hierarchical).
        label_mode: ``'binary'`` or ``'hierarchical'``.
    """

    def __init__(self, input_dim: int, num_classes: int, label_mode: str = "binary"):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.label_mode = label_mode

    def forward(self, x):
        """Should be implemented by subclasses.

        Args:
            x: Tensor of shape ``(batch_size, time_steps, input_dim)``.

        Returns:
            A dict with at least ``"logits"`` of shape
            ``(batch_size, num_classes)``.
        """
        raise NotImplementedError

    def _compute_probs_preds(self, logits: torch.Tensor) -> dict:
        """Compute probabilities and predictions from logits.

        Handles both binary (softmax) and hierarchical (sigmoid) modes.

        Args:
            logits: Raw logits of shape ``(B, num_classes)``.

        Returns:
            Dict with ``"logits"``, ``"probs"``, ``"preds"``.
        """
        if self.label_mode == "hierarchical":
            probs = torch.sigmoid(logits)
            # Primary prediction uses bit 1 (unhealthy indicator)
            preds = (probs[:, 1] > 0.5).long()
        else:
            probs = F.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)
        return {"logits": logits, "probs": probs, "preds": preds}

    def compute_loss(self, x, y):
        """Compute loss given input sequences and targets.

        Args:
            x: Tensor of shape ``(batch_size, time_steps, input_dim)``.
            y: LongTensor of shape ``(batch_size,)`` with class indices.

        Returns:
            Dict with ``"loss"`` and all keys from ``forward()``.
        """
        outputs = self(x)
        logits = outputs["logits"]
        loss = F.cross_entropy(logits, y)
        return {"loss": loss, **outputs}


# ---------------------------------------------------------------------
# 1. Simple LSTM classifier
# ---------------------------------------------------------------------
class LSTMClassifier(BaseTimeSeriesClassifier):
    """
    Simple LSTM-based time series classifier:
    LSTM encoder -> last hidden state -> MLP classifier.
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.1,
        pooling: str = "last",
        mlp_multiplier: float = 2.0,
        use_layer_norm: bool = True,
        attention_pool: bool = False,
        label_mode: str = "binary",
    ):
        super().__init__(input_dim, num_classes, label_mode=label_mode)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pooling = pooling.lower()
        self.use_layer_norm = use_layer_norm
        self.use_attention_pool = attention_pool

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)

        if attention_pool:
            self.attn_pooling = SegmentAttentionPooling(lstm_out_dim)
            feature_dim = lstm_out_dim
        elif self.pooling in ("mean_max", "concat"):
            feature_dim = lstm_out_dim * 2
        else:
            feature_dim = lstm_out_dim

        hidden_fc = max(int(feature_dim * mlp_multiplier), feature_dim)
        layers = []
        if self.use_layer_norm:
            layers.append(nn.LayerNorm(feature_dim))
        layers.extend([
            nn.Linear(feature_dim, hidden_fc),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_fc, num_classes),
        ])
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)

        result = {}
        if self.use_attention_pool:
            features, attn_weights = self.attn_pooling(lstm_out)
            result["attn_weights"] = attn_weights
        elif self.pooling == "mean":
            features = lstm_out.mean(dim=1)
        elif self.pooling == "max":
            features, _ = torch.max(lstm_out, dim=1)
        elif self.pooling == "mean_max":
            mean_val = lstm_out.mean(dim=1)
            max_val, _ = torch.max(lstm_out, dim=1)
            features = torch.cat([mean_val, max_val], dim=1)
        elif self.pooling == "concat":
            last_state = lstm_out[:, -1, :]
            mean_state = lstm_out.mean(dim=1)
            features = torch.cat([last_state, mean_state], dim=1)
        else:
            features = lstm_out[:, -1, :]

        logits = self.classifier(features)
        result.update(self._compute_probs_preds(logits))
        return result


# ---------------------------------------------------------------------
# 2. 1D CNN classifier
# ---------------------------------------------------------------------
class CNN1DClassifier(BaseTimeSeriesClassifier):
    """
    1D CNN over the time dimension with global max pooling.
    Input (B, T, D) is treated as D channels and sequence length T.
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_filters: int = 64,
        kernel_sizes=(3, 5, 7),
        dropout: float = 0.1,
        label_mode: str = "binary",
    ):
        super().__init__(input_dim, num_classes, label_mode=label_mode)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=input_dim, out_channels=num_filters,
                      kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)

        conv_outs = []
        for conv in self.convs:
            h = F.relu(conv(x))
            h = F.adaptive_max_pool1d(h, 1).squeeze(-1)
            conv_outs.append(h)

        features = self.dropout(torch.cat(conv_outs, dim=1))
        logits = self.fc(features)
        return self._compute_probs_preds(logits)


# ---------------------------------------------------------------------
# 3. CNN-LSTM hybrid classifier
# ---------------------------------------------------------------------
class CNNLSTMClassifier(BaseTimeSeriesClassifier):
    """
    CNN-LSTM hybrid classifier for time series classification.

    Architecture:
        1. Multi-kernel CNN extracts local multi-scale patterns from VAE latent trajectories
        2. BiLSTM captures bidirectional temporal context
        3. Configurable pooling aggregates over time
        4. MLP classification head

    Input: (B, T, D) — T timesteps of D-dim VAE latent features
    Output: dict with "logits", "probs", "preds"
    """
    def __init__(
        self,
        input_dim: int = 16,
        num_classes: int = 2,
        num_filters: int = 32,
        kernel_sizes: Sequence[int] = (3, 5, 7),
        cnn_out_dim: int = 64,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        pooling: str = "mean_max",
        mlp_multiplier: float = 2.0,
        use_layer_norm: bool = True,
        attention_pool: bool = False,
        label_mode: str = "binary",
    ):
        super().__init__(input_dim, num_classes, label_mode=label_mode)
        self.pooling = pooling.lower()
        self.use_attention_pool = attention_pool

        # --- CNN feature extraction (parallel multi-kernel) ---
        self.conv_branches = nn.ModuleList()
        for k in kernel_sizes:
            self.conv_branches.append(nn.Sequential(
                nn.Conv1d(input_dim, num_filters, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(num_filters),
                nn.GELU(),
            ))

        concat_filters = num_filters * len(kernel_sizes)
        self.cnn_projection = nn.Sequential(
            nn.Conv1d(concat_filters, cnn_out_dim, kernel_size=1),
            nn.BatchNorm1d(cnn_out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # --- BiLSTM temporal modeling ---
        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        lstm_out_dim = lstm_hidden * 2  # bidirectional
        self.lstm_norm = nn.LayerNorm(lstm_out_dim) if use_layer_norm else nn.Identity()

        # --- Pooling ---
        if attention_pool:
            self.attn_pooling = SegmentAttentionPooling(lstm_out_dim)
            pooled_dim = lstm_out_dim
        elif self.pooling in ("mean_max", "concat"):
            pooled_dim = lstm_out_dim * 2
        else:
            pooled_dim = lstm_out_dim

        # --- FC classification head ---
        hidden_fc = max(int(pooled_dim * mlp_multiplier), pooled_dim)
        head_layers = []
        if use_layer_norm:
            head_layers.append(nn.LayerNorm(pooled_dim))
        head_layers.extend([
            nn.Linear(pooled_dim, hidden_fc),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_fc, num_classes),
        ])
        self.classifier = nn.Sequential(*head_layers)

    def forward(self, x):
        h = x.transpose(1, 2)

        branch_outs = [branch(h) for branch in self.conv_branches]
        h = torch.cat(branch_outs, dim=1)
        h = self.cnn_projection(h).transpose(1, 2)

        lstm_out, _ = self.lstm(h)
        lstm_out = self.lstm_norm(lstm_out)

        result = {}
        if self.use_attention_pool:
            features, attn_weights = self.attn_pooling(lstm_out)
            result["attn_weights"] = attn_weights
        elif self.pooling == "mean":
            features = lstm_out.mean(dim=1)
        elif self.pooling == "max":
            features, _ = torch.max(lstm_out, dim=1)
        elif self.pooling == "mean_max":
            mean_val = lstm_out.mean(dim=1)
            max_val, _ = torch.max(lstm_out, dim=1)
            features = torch.cat([mean_val, max_val], dim=1)
        elif self.pooling == "concat":
            features = torch.cat([lstm_out[:, -1, :], lstm_out.mean(dim=1)], dim=1)
        else:
            features = lstm_out[:, -1, :]

        logits = self.classifier(features)
        result.update(self._compute_probs_preds(logits))
        return result


# ---------------------------------------------------------------------
# 4. BiLSTM + self-attention classifier
# ---------------------------------------------------------------------
class BiLSTMAttentionClassifier(BaseTimeSeriesClassifier):
    """
    BiLSTM with self-attention pooling over the time dimension.

    H_t = BiLSTM(x)_t
    a_t = softmax(v^T tanh(W H_t))
    z = sum_t a_t H_t
    logits = MLP(z)
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        attn_dim: int = 64,
        dropout: float = 0.1,
        label_mode: str = "binary",
    ):
        super().__init__(input_dim, num_classes, label_mode=label_mode)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_out_dim = hidden_dim * 2

        self.attn = nn.Linear(lstm_out_dim, attn_dim)
        self.attn_vector = nn.Parameter(torch.randn(attn_dim))

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, x):
        H, _ = self.lstm(x)

        attn_scores = torch.tanh(self.attn(H))
        attn_scores = torch.matmul(attn_scores, self.attn_vector)
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)

        context = (H * attn_weights).sum(dim=1)
        context = self.dropout(context)

        logits = self.fc(context)
        result = self._compute_probs_preds(logits)
        result["attn_weights"] = attn_weights.squeeze(-1)
        return result


# ---------------------------------------------------------------------
# 4. Transformer encoder classifier
# ---------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, T, d_model)
        """
        T = x.size(1)
        return x + self.pe[:, :T, :]


class TransformerClassifier(BaseTimeSeriesClassifier):
    """
    Transformer encoder for time series classification.

    Input -> linear projection -> positional encoding ->
    TransformerEncoder -> pooled representation -> classifier.
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 5000,
        pooling: str = "mean",
        label_mode: str = "binary",
    ):
        super().__init__(input_dim, num_classes, label_mode=label_mode)

        self.d_model = d_model
        self.pooling = pooling

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        else:
            self.cls_token = None

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x, src_key_padding_mask=None):
        B, T, D = x.shape
        h = self.input_proj(x)

        if self.pooling == "cls":
            cls_tokens = self.cls_token.expand(B, -1, -1)
            h = torch.cat([cls_tokens, h], dim=1)
            if src_key_padding_mask is not None:
                cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=src_key_padding_mask.device)
                src_key_padding_mask = torch.cat([cls_pad, src_key_padding_mask], dim=1)

        h = self.pos_encoding(h)
        encoded = self.encoder(h, src_key_padding_mask=src_key_padding_mask)

        if self.pooling == "cls":
            features = encoded[:, 0, :]
        else:
            if src_key_padding_mask is not None:
                mask = (~src_key_padding_mask).unsqueeze(-1)
                features = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                features = encoded.mean(dim=1)

        logits = self.fc(self.dropout(features))
        return self._compute_probs_preds(logits)

# ---------------------------------------------------------------------
# 5. Mamba (Selective State Space) classifier — pure PyTorch
# ---------------------------------------------------------------------
class SelectiveSSM(nn.Module):
    """
    Pure-PyTorch selective state space block inspired by Mamba (Gu & Dao 2023).
    Uses input-dependent discretisation so the model learns *what* to remember.

    Core idea:  x -> depthwise conv -> split into (gate, ssm_input)
                ssm_input -> compute B,C,delta from input -> discretise A,B -> scan -> output * gate
    """
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2, conv_kernel: int = 4, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        d_inner = d_model * expand
        self.d_inner = d_inner

        # Input projection: d_model -> 2 * d_inner (for gate + ssm_input)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)

        # Depthwise conv over time for local context
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=conv_kernel,
            padding=conv_kernel - 1, groups=d_inner, bias=True,
        )

        # SSM parameters from input
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)  # B, C, delta

        # Learnable log(A) initialised to HiPPO-style values
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))  # (d_inner, d_state)

        self.D = nn.Parameter(torch.ones(d_inner))  # skip connection

        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) -> (B, T, d_model)"""
        B, T, _ = x.shape

        # Project and split
        xz = self.in_proj(x)                         # (B, T, 2*d_inner)
        x_ssm, z = xz.chunk(2, dim=-1)               # each (B, T, d_inner)

        # Local conv (causal: trim future)
        x_ssm = x_ssm.transpose(1, 2)                # (B, d_inner, T)
        x_ssm = self.conv1d(x_ssm)[:, :, :T]         # causal trim
        x_ssm = x_ssm.transpose(1, 2)                # (B, T, d_inner)
        x_ssm = F.silu(x_ssm)

        # Compute input-dependent B, C, delta
        bcdt = self.x_proj(x_ssm)                     # (B, T, 2*d_state+1)
        B_inp = bcdt[..., :self.d_state]               # (B, T, d_state)
        C_inp = bcdt[..., self.d_state:2*self.d_state] # (B, T, d_state)
        delta = F.softplus(bcdt[..., -1])              # (B, T)

        # Discretise: A_bar = exp(delta * A), B_bar = delta * B
        A = -torch.exp(self.A_log)                     # (d_inner, d_state)
        delta_u = delta.unsqueeze(-1)                  # (B, T, 1)
        A_bar = torch.exp(delta_u.unsqueeze(-1) * A)   # (B, T, d_inner, d_state)
        B_bar = delta_u.unsqueeze(-1) * B_inp.unsqueeze(2)  # (B, T, 1, d_state) broadcast

        # Sequential scan
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            h = A_bar[:, t] * h + B_bar[:, t] * x_ssm[:, t].unsqueeze(-1)  # (B, d_inner, d_state)
            y_t = (h * C_inp[:, t].unsqueeze(1)).sum(dim=-1)                # (B, d_inner)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)  # (B, T, d_inner)

        # Gate + skip + project
        y = y + x_ssm * self.D
        y = y * F.silu(z)
        y = self.out_proj(y)
        y = self.dropout(y)
        return y


class MambaBlock(nn.Module):
    """Single Mamba block: LayerNorm -> SelectiveSSM -> residual."""
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2,
                 conv_kernel: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, expand, conv_kernel, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ssm(self.norm(x))


class MambaClassifier(BaseTimeSeriesClassifier):
    """
    Mamba-based time series classifier using selective state space models.

    Advantages over Transformers for this task:
    - Linear O(T) complexity vs O(T^2) attention
    - Input-dependent gating learns which time points matter
    - Lightweight (~50-80K params with default settings)

    Architecture:
        Linear(input_dim -> d_model) -> [MambaBlock x n_layers] -> pooling -> MLP head
    """
    def __init__(
        self,
        input_dim: int = 16,
        num_classes: int = 2,
        d_model: int = 64,
        d_state: int = 16,
        expand: int = 2,
        conv_kernel: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        pooling: str = "mean_max",
        mlp_multiplier: float = 2.0,
        attention_pool: bool = False,
        label_mode: str = "binary",
    ):
        super().__init__(input_dim, num_classes, label_mode=label_mode)
        self.pooling = pooling.lower()
        self.use_attention_pool = attention_pool

        self.input_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, expand, conv_kernel, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        if attention_pool:
            self.attn_pooling = SegmentAttentionPooling(d_model)
            pooled_dim = d_model
        elif self.pooling in ("mean_max", "concat"):
            pooled_dim = d_model * 2
        else:
            pooled_dim = d_model

        hidden_fc = max(int(pooled_dim * mlp_multiplier), pooled_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hidden_fc),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_fc, num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        h = self.final_norm(h)

        result = {}
        if self.use_attention_pool:
            features, attn_weights = self.attn_pooling(h)
            result["attn_weights"] = attn_weights
        elif self.pooling == "mean":
            features = h.mean(dim=1)
        elif self.pooling == "max":
            features, _ = torch.max(h, dim=1)
        elif self.pooling == "mean_max":
            features = torch.cat([h.mean(dim=1), torch.max(h, dim=1)[0]], dim=1)
        elif self.pooling == "concat":
            features = torch.cat([h[:, -1, :], h.mean(dim=1)], dim=1)
        else:
            features = h[:, -1, :]

        logits = self.classifier(features)
        result.update(self._compute_probs_preds(logits))
        return result


# ---------------------------------------------------------------------
# 6. Multi-Scale Conv + Squeeze-Excitation + Attention classifier
# ---------------------------------------------------------------------
class SqueezeExcitation(nn.Module):
    """Channel attention: learn which feature channels are important."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        w = x.mean(dim=-1)          # (B, C)  global avg pool over time
        w = self.fc(w).unsqueeze(-1) # (B, C, 1)
        return x * w


class InceptionBlock(nn.Module):
    """
    Multi-scale 1D convolution block inspired by InceptionTime.
    Parallel branches with different kernel sizes capture patterns at
    different temporal scales (beat-to-beat, decelerations, baseline shifts).
    """
    def __init__(self, in_channels: int, num_filters: int, kernel_sizes: Sequence[int],
                 use_residual: bool = True):
        super().__init__()
        # Bottleneck: reduce channels before expensive convs
        bottleneck_dim = max(in_channels // 4, 1)
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_dim, kernel_size=1, bias=False)

        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            self.branches.append(nn.Sequential(
                nn.Conv1d(bottleneck_dim, num_filters, kernel_size=k, padding=k // 2, bias=False),
                nn.BatchNorm1d(num_filters),
            ))
        # Max-pool branch
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, num_filters, kernel_size=1, bias=False),
            nn.BatchNorm1d(num_filters),
        )

        out_channels = num_filters * len(kernel_sizes) + num_filters  # conv branches + pool
        self.bn_out = nn.BatchNorm1d(out_channels)

        # Residual
        self.use_residual = use_residual
        if use_residual:
            self.residual = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            ) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T)
        bn = self.bottleneck(x)
        outs = [branch(bn) for branch in self.branches]
        outs.append(self.pool_branch(x))
        h = torch.cat(outs, dim=1)  # (B, out_channels, T)
        h = self.bn_out(h)
        if self.use_residual:
            h = h + self.residual(x)
        return F.gelu(h)


class MultiScaleConvAttentionClassifier(BaseTimeSeriesClassifier):
    """
    Multi-scale convolutional classifier with squeeze-excitation and self-attention.

    Architecture:
        Transpose(B,T,D -> B,D,T)
        -> [InceptionBlock x n_blocks] (multi-scale temporal feature extraction)
        -> SqueezeExcitation (channel attention — which latent dims matter)
        -> Multi-Head Self-Attention (global temporal reasoning)
        -> Adaptive mean+max pool
        -> MLP classification head

    Designed for VAE latent trajectories (B, 300, 16) where FHR patterns
    operate at multiple temporal scales simultaneously.
    """
    def __init__(
        self,
        input_dim: int = 16,
        num_classes: int = 2,
        num_filters: int = 32,
        kernel_sizes: Sequence[int] = (5, 19, 39),
        n_inception_blocks: int = 2,
        se_reduction: int = 8,
        n_attn_heads: int = 4,
        attn_dropout: float = 0.1,
        dropout: float = 0.1,
        mlp_multiplier: float = 2.0,
        label_mode: str = "binary",
    ):
        super().__init__(input_dim, num_classes, label_mode=label_mode)

        blocks = []
        in_ch = input_dim
        for i in range(n_inception_blocks):
            block = InceptionBlock(in_ch, num_filters, kernel_sizes, use_residual=True)
            in_ch = num_filters * len(kernel_sizes) + num_filters
            blocks.append(block)
        self.inception_stack = nn.Sequential(*blocks)
        feat_dim = in_ch

        self.se = SqueezeExcitation(feat_dim, reduction=se_reduction)

        self.attn_norm = nn.LayerNorm(feat_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=feat_dim, num_heads=n_attn_heads,
            dropout=attn_dropout, batch_first=True,
        )

        pooled_dim = feat_dim * 2

        hidden_fc = max(int(pooled_dim * mlp_multiplier), pooled_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hidden_fc),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_fc, num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict:
        h = x.transpose(1, 2)
        h = self.se(self.inception_stack(h))
        h = h.transpose(1, 2)
        h_attn, _ = self.self_attn(self.attn_norm(h), self.attn_norm(h), self.attn_norm(h))
        h = h + h_attn

        h_t = h.transpose(1, 2)
        features = torch.cat([
            F.adaptive_avg_pool1d(h_t, 1).squeeze(-1),
            F.adaptive_max_pool1d(h_t, 1).squeeze(-1),
        ], dim=1)

        logits = self.classifier(features)
        return self._compute_probs_preds(logits)


# ---------------------------------------------------------------------
# 7. Causal CNN-LSTM classifier (depthwise separable, dilated, residual)
# ---------------------------------------------------------------------
class CausalConvBlock(nn.Module):
    """Depthwise separable causal convolution block with residual connection.

    Causal: output at time *t* depends only on inputs at times <= *t*.
    Depthwise: per-channel temporal convolution (efficient, avoids cross-channel
    mixing).  Pointwise: 1x1 conv for channel mixing and expansion.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Temporal kernel size for depthwise conv.
        dilation: Dilation factor (controls receptive-field growth).
        dropout: Dropout probability applied after pointwise conv.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.causal_pad = (kernel_size - 1) * dilation

        # Depthwise causal conv
        self.dw_conv = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            dilation=dilation, groups=in_channels, bias=False,
        )
        self.bn1 = nn.BatchNorm1d(in_channels)

        # Pointwise (1x1) for channel expansion / mixing
        self.pw_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Residual projection (1x1 if channel mismatch)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape ``(B, C_in, T)``.

        Returns:
            Tensor of shape ``(B, C_out, T)``.
        """
        res = self.residual(x)

        h = F.pad(x, (self.causal_pad, 0))  # causal left-pad
        h = self.dw_conv(h)                   # (B, C_in, T)
        h = self.bn1(h)
        h = F.gelu(h)

        h = self.pw_conv(h)                   # (B, C_out, T)
        h = self.bn2(h)
        h = F.gelu(h)
        h = self.dropout(h)

        return h + res


class CausalCNNLSTMClassifier(BaseTimeSeriesClassifier):
    """Causal CNN-LSTM classifier with depthwise separable dilated convolutions.

    Architecture::

        Input (B, T, D) -> transpose (B, D, T)
          -> [CausalConvBlock_1: D  -> C1, dilation=d1]
          -> [CausalConvBlock_2: C1 -> C2, dilation=d2]
          -> [CausalConvBlock_3: C2 -> C3, dilation=d3]
          -> transpose (B, T, C3)
          -> BiLSTM (C3 -> 2*lstm_hidden)
          -> LayerNorm
          -> Pooling (mean_max -> 4*lstm_hidden)
          -> MLP Head -> logits (B, num_classes)

    Args:
        input_dim: Feature dimension of each timestep.
        num_classes: Number of output classes.
        conv_channels: Channel count for each causal conv stage.
        kernel_sizes: Kernel size per stage.
        dilations: Dilation factor per stage.
        lstm_hidden: LSTM hidden dim per direction (bidirectional).
        lstm_layers: Number of stacked BiLSTM layers.
        dropout: Dropout probability.
        pooling: Pooling strategy (``mean_max``, ``mean``, ``max``, ``last``,
            ``concat``).
        mlp_multiplier: MLP hidden dim = pooled_dim * multiplier.
    """

    def __init__(
        self,
        input_dim: int = 16,
        num_classes: int = 2,
        conv_channels: Sequence[int] = (32, 64, 128),
        kernel_sizes: Sequence[int] = (5, 7, 11),
        dilations: Sequence[int] = (1, 2, 4),
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        pooling: str = "mean_max",
        mlp_multiplier: float = 2.0,
        attention_pool: bool = False,
        label_mode: str = "binary",
    ):
        super().__init__(input_dim, num_classes, label_mode=label_mode)
        self.pooling = pooling.lower()
        self.use_attention_pool = attention_pool

        # --- Sequential causal conv stages ---
        conv_blocks = []
        in_ch = input_dim
        for ch, ks, dil in zip(conv_channels, kernel_sizes, dilations):
            conv_blocks.append(CausalConvBlock(in_ch, ch, ks, dilation=dil, dropout=dropout))
            in_ch = ch
        self.conv_stages = nn.Sequential(*conv_blocks)

        # --- BiLSTM temporal modelling ---
        self.lstm = nn.LSTM(
            input_size=in_ch,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        lstm_out_dim = lstm_hidden * 2  # bidirectional
        self.lstm_norm = nn.LayerNorm(lstm_out_dim)

        # --- Pooling ---
        if attention_pool:
            self.attn_pooling = SegmentAttentionPooling(lstm_out_dim)
            pooled_dim = lstm_out_dim
        elif self.pooling in ("mean_max", "concat"):
            pooled_dim = lstm_out_dim * 2
        else:
            pooled_dim = lstm_out_dim

        # --- MLP classification head ---
        hidden_fc = max(int(pooled_dim * mlp_multiplier), pooled_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hidden_fc),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_fc, num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass.

        Args:
            x: Tensor of shape ``(B, T, D)``.

        Returns:
            Dict with ``logits`` ``(B, num_classes)``, ``probs``, ``preds``.
        """
        # (B, T, D) -> (B, D, T) for Conv1d
        h = x.transpose(1, 2)

        # Sequential causal conv stages
        h = self.conv_stages(h)  # (B, C_last, T)

        # (B, C_last, T) -> (B, T, C_last) for LSTM
        h = h.transpose(1, 2)

        # BiLSTM
        lstm_out, _ = self.lstm(h)  # (B, T, lstm_out_dim)
        lstm_out = self.lstm_norm(lstm_out)

        # Pooling
        result = {}
        if self.use_attention_pool:
            features, attn_weights = self.attn_pooling(lstm_out)
            result["attn_weights"] = attn_weights
        elif self.pooling == "mean":
            features = lstm_out.mean(dim=1)
        elif self.pooling == "max":
            features, _ = torch.max(lstm_out, dim=1)
        elif self.pooling == "mean_max":
            mean_val = lstm_out.mean(dim=1)
            max_val, _ = torch.max(lstm_out, dim=1)
            features = torch.cat([mean_val, max_val], dim=1)
        elif self.pooling == "concat":
            last_state = lstm_out[:, -1, :]
            mean_state = lstm_out.mean(dim=1)
            features = torch.cat([last_state, mean_state], dim=1)
        else:
            features = lstm_out[:, -1, :]

        logits = self.classifier(features)
        result.update(self._compute_probs_preds(logits))
        return result


# ---------------------------------------------------------------------
# TLO (Time from Labour Onset) Embedding
# ---------------------------------------------------------------------
class TLOEmbedding(nn.Module):
    """Embeds scalar Time from Labour Onset into a learned vector.

    NaN values (unavailable TLO) are replaced by a learned
    ``missing_embedding`` parameter so the model can distinguish
    "unknown TLO" from any real value.

    Args:
        embed_dim: Dimensionality of the output embedding.
        dropout: Dropout probability inside the MLP.
    """

    def __init__(self, embed_dim: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.missing_embedding = nn.Parameter(torch.zeros(embed_dim))
        self.mlp = nn.Sequential(
            nn.Linear(1, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, tlo_seconds: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Forward pass.

        Args:
            tlo_seconds: Scalar TLO per sample ``(B,)`` in seconds.
                May contain ``NaN`` when TLO is unavailable.
            seq_len: Temporal dimension *T* to broadcast to (e.g. 300).

        Returns:
            Embedding tensor of shape ``(B, seq_len, embed_dim)``.
        """
        is_valid = ~torch.isnan(tlo_seconds)
        tlo_hours = tlo_seconds / 3600.0
        tlo_hours = torch.where(is_valid, tlo_hours, torch.zeros_like(tlo_hours))

        tlo_embed = self.mlp(tlo_hours.unsqueeze(-1))  # (B, embed_dim)

        missing_mask = (~is_valid).unsqueeze(-1)  # (B, 1)
        tlo_embed = torch.where(
            missing_mask,
            self.missing_embedding.unsqueeze(0).expand_as(tlo_embed),
            tlo_embed,
        )

        return tlo_embed.unsqueeze(1).expand(-1, seq_len, -1)  # (B, T, embed_dim)


class VaeTebTimeSeriesClassifier(nn.Module):
    """Combined VAE + Classifier model for time series classification.

    This model uses a pre-trained VAE (SeqVae) to encode time series data into
    latent representations, optionally enriches them with transfer entropy
    signals, concatenates a TLO embedding, then classifies the result.

    Architecture::

        VAE Encoder -> encode_only() outputs
          -> [enriched: mu_post || logvar_post || residual || kld]  (B, T, 64)
          -> [or plain: mu_post]                                   (B, T, 16)
          -> [augmentation: posterior noise + temporal jitter]      (training only)
          -> [TLO scalar -> TLOEmbedding -> (B, T, tlo_embed_dim)] (optional)
          -> concat -> Classifier -> logits (B, num_classes)

    Args:
        vae_model: Pre-trained SeqVae model for encoding.
        classifier: Any classifier from ``BaseTimeSeriesClassifier``.
        freeze_vae: If ``True``, freeze VAE weights during training.
        use_posterior: If ``True``, use posterior z from q(z|x,y); else prior.
        sample_latent: If ``True``, sample from latent distribution; else mean.
        class_weights: Optional per-class weights for cross-entropy loss.
        tlo_embed_dim: TLO embedding dimension.  ``0`` disables TLO.
        tlo_dropout: Dropout inside TLO embedding MLP.
        enriched_features: If ``True``, concatenate KLD, logvar_post, and
            posterior-prior residual to mu_post (16-dim → 64-dim).
        label_mode: ``'binary'`` for 2-class softmax or ``'hierarchical'``
            for 3-bit multi-hot sigmoid.
        focal_gamma: Focal loss gamma parameter.  ``0`` disables focusing.
        label_smoothing: Label smoothing for focal/BCE loss.
        bit_weights: Per-bit loss weights for hierarchical mode ``(3,)``.
        augment_posterior_sample: Add posterior noise during training.
        augment_noise_scale: Scale factor for posterior noise.
        augment_temporal_jitter: Max random temporal shift (timesteps).
    """

    def __init__(
        self,
        vae_model: SeqVae,
        classifier: BaseTimeSeriesClassifier,
        freeze_vae: bool = True,
        use_posterior: bool = True,
        sample_latent: bool = False,
        class_weights: Optional[Sequence[float]] = None,
        tlo_embed_dim: int = 0,
        tlo_dropout: float = 0.1,
        enriched_features: bool = False,
        label_mode: str = "binary",
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
        bit_weights: Optional[Sequence[float]] = None,
        augment_posterior_sample: bool = False,
        augment_noise_scale: float = 0.5,
        augment_temporal_jitter: int = 0,
    ):
        super().__init__()
        self.vae_model = vae_model
        self.classifier = classifier
        self.freeze_vae = freeze_vae
        self.use_posterior = use_posterior
        self.sample_latent = sample_latent
        self.enriched_features = enriched_features
        self.label_mode = label_mode

        # Augmentation config
        self.augment_posterior_sample = augment_posterior_sample
        self.augment_noise_scale = augment_noise_scale
        self.augment_temporal_jitter = augment_temporal_jitter

        # Class weights (for binary mode backward compat)
        if class_weights is not None:
            weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", weight_tensor)
        else:
            self.class_weights = None

        # Focal loss (used in hierarchical mode, also available for binary)
        bw_tensor = None
        if bit_weights is not None:
            bw_tensor = torch.as_tensor(bit_weights, dtype=torch.float32)
        self.focal_loss = FocalBCEWithLogitsLoss(
            gamma=focal_gamma,
            alpha=bw_tensor,
            label_smoothing=label_smoothing,
        )

        # TLO embedding (disabled when tlo_embed_dim == 0)
        if tlo_embed_dim > 0:
            self.tlo_embedding = TLOEmbedding(embed_dim=tlo_embed_dim, dropout=tlo_dropout)
        else:
            self.tlo_embedding = None

        if self.freeze_vae:
            for param in self.vae_model.parameters():
                param.requires_grad = False
            self.vae_model.eval()

    def _run_vae_encoder(
        self,
        y_st: torch.Tensor,
        y_ph: torch.Tensor,
        x_ph: torch.Tensor,
    ) -> dict:
        """Run VAE encoder and return all outputs.

        Args:
            y_st: Target scattering features ``(B, T, 43)``.
            y_ph: Target phase harmonic features ``(B, T, 44)``.
            x_ph: Source cross-phase features ``(B, T, 137)``.

        Returns:
            Dict from ``vae_model.encode_only()`` containing
            ``mu_post``, ``logvar_post``, ``mu_prior``, ``logvar_prior``, ``z``.
        """
        if self.freeze_vae:
            self.vae_model.eval()
            with torch.no_grad():
                return self.vae_model.encode_only(
                    y_st=y_st, y_ph=y_ph, x_ph=x_ph,
                    sample_z=self.sample_latent,
                )
        return self.vae_model.encode_only(
            y_st=y_st, y_ph=y_ph, x_ph=x_ph,
            sample_z=self.sample_latent,
        )

    @staticmethod
    def compute_kld_per_dim(
        mu_post: torch.Tensor,
        logvar_post: torch.Tensor,
        mu_prior: torch.Tensor,
        logvar_prior: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-dimension KL divergence (transfer entropy signal).

        KL(q(z|x,y) || p(z|y)) per dimension per timestep.

        Args:
            mu_post: Posterior mean ``(B, T, D)``.
            logvar_post: Posterior log-variance ``(B, T, D)``.
            mu_prior: Prior mean ``(B, T, D)``.
            logvar_prior: Prior log-variance ``(B, T, D)``.

        Returns:
            Per-dim KLD of shape ``(B, T, D)``.
        """
        var_post = logvar_post.exp()
        var_prior = logvar_prior.exp().clamp(min=1e-8)
        return 0.5 * (
            logvar_prior - logvar_post
            + var_post / var_prior
            + (mu_post - mu_prior).pow(2) / var_prior
            - 1.0
        )

    def encode_features(
        self,
        y_st: torch.Tensor,
        y_ph: torch.Tensor,
        x_ph: torch.Tensor,
    ) -> torch.Tensor:
        """Extract latent features from VAE encoder.

        When ``enriched_features=True``, concatenates four 16-dim signals
        into a 64-dim feature vector per timestep:

        1. ``mu_post`` — posterior mean
        2. ``logvar_post`` — posterior uncertainty
        3. ``mu_post - mu_prior`` — directed transfer residual
        4. ``kld_per_dim`` — per-dimension transfer entropy

        When ``enriched_features=False``, returns only ``mu_post`` (16-dim).

        Training-time augmentation (posterior noise, temporal jitter) is
        applied to ``mu_post`` before enrichment.

        Args:
            y_st: Target scattering features ``(B, T, 43)``.
            y_ph: Target phase harmonic features ``(B, T, 44)``.
            x_ph: Source cross-phase + UP self-phase features ``(B, T, 137)``.

        Returns:
            Feature tensor ``(B, T, D)`` where D is 64 (enriched) or 16.
        """
        enc = self._run_vae_encoder(y_st, y_ph, x_ph)

        mu_post = enc["mu_post"]
        logvar_post = enc["logvar_post"]
        mu_prior = enc["mu_prior"]
        logvar_prior = enc["logvar_prior"]

        # --- Training-time augmentation on mu_post ---
        z = mu_post
        if self.use_posterior:
            if self.sample_latent:
                z = enc["z"]
        else:
            z = mu_prior

        if self.training:
            if self.augment_posterior_sample:
                noise = torch.randn_like(z) * (0.5 * logvar_post).exp() * self.augment_noise_scale
                z = z + noise

            if self.augment_temporal_jitter > 0:
                shift = torch.randint(
                    -self.augment_temporal_jitter,
                    self.augment_temporal_jitter + 1,
                    (1,),
                ).item()
                if shift != 0:
                    z = torch.roll(z, shifts=shift, dims=1)

        if not self.enriched_features:
            return z

        # --- Enriched: concatenate 4 signals ---
        residual = mu_post - mu_prior
        kld = self.compute_kld_per_dim(mu_post, logvar_post, mu_prior, logvar_prior)
        return torch.cat([z, logvar_post, residual, kld], dim=-1)

    def forward(
        self,
        y_st: torch.Tensor,
        y_ph: torch.Tensor,
        x_ph: torch.Tensor,
        tlo: Optional[torch.Tensor] = None,
    ) -> dict:
        """Forward pass through VAE encoder + optional TLO embedding + classifier.

        Args:
            y_st: Target scattering features ``(B, T, 43)``.
            y_ph: Target phase harmonic features ``(B, T, 44)``.
            x_ph: Source cross-phase + UP self-phase features ``(B, T, 137)``.
            tlo: Optional scalar TLO per sample ``(B,)`` in seconds (may
                contain ``NaN``).

        Returns:
            Dictionary with ``logits``, ``probs``, ``preds``,
            ``latent_features``.
        """
        z = self.encode_features(y_st, y_ph, x_ph)  # (B, T, D)

        # Concatenate TLO embedding if enabled
        if self.tlo_embedding is not None:
            if tlo is None:
                tlo = torch.full((z.shape[0],), float('nan'), device=z.device)
            tlo_embed = self.tlo_embedding(tlo, seq_len=z.shape[1])
            z = torch.cat([z, tlo_embed], dim=-1)

        classifier_outputs = self.classifier(z)
        classifier_outputs["latent_features"] = z

        return classifier_outputs

    def compute_loss(
        self,
        y_st: torch.Tensor,
        y_ph: torch.Tensor,
        x_ph: torch.Tensor,
        labels: torch.Tensor,
        tlo: Optional[torch.Tensor] = None,
    ) -> dict:
        """Compute classification loss.

        Supports both binary (cross-entropy) and hierarchical (focal BCE)
        modes, controlled by ``self.label_mode``.

        Args:
            y_st: Target scattering features ``(B, T, 43)``.
            y_ph: Target phase harmonic features ``(B, T, 44)``.
            x_ph: Source cross-phase + UP self-phase features ``(B, T, 137)``.
            labels: Ground truth class labels ``(B,)``.
            tlo: Optional scalar TLO per sample ``(B,)`` in seconds.

        Returns:
            Dictionary with ``loss``, ``accuracy``, ``logits``, ``probs``,
            ``preds``.
        """
        outputs = self.forward(y_st, y_ph, x_ph, tlo=tlo)
        logits = outputs["logits"]

        if self.label_mode == "hierarchical":
            hier_targets = map_to_hierarchical_labels(labels)
            loss = self.focal_loss(logits, hier_targets)
            # Primary metric: unhealthy detection (bit 1)
            unhealthy_prob = torch.sigmoid(logits[:, 1])
            preds = (unhealthy_prob > 0.5).long()
            binary_labels = (labels > 1).long()
        else:
            loss = F.cross_entropy(
                logits, labels, weight=self.class_weights,
            )
            preds = outputs["preds"]
            binary_labels = labels

        accuracy = (preds == binary_labels).float().mean()

        return {
            "loss": loss,
            "accuracy": accuracy,
            **outputs,
        }
