try:
    # Preferred (package) import
    from .vae_teb_model_prediction import *  # noqa: F403
except ImportError:
    # Backward-compatible fallback when running from this directory
    from vae_teb_model_prediction import *  # noqa: F403

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence


class BaseTimeSeriesClassifier(nn.Module):
    """
    Base class that implements a standard compute_loss for
    multi-class classification.
    """
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

    def forward(self, x):
        """
        Should be implemented by subclasses.

        Args:
            x: Tensor of shape (batch_size, time_steps, input_dim)

        Returns:
            A dict with at least the key "logits" of shape (batch_size, num_classes).
        """
        raise NotImplementedError

    def compute_loss(self, x, y):
        """
        Compute cross-entropy loss given input sequences and targets.

        Args:
            x: Tensor of shape (batch_size, time_steps, input_dim)
            y: LongTensor of shape (batch_size,) with class indices.

        Returns:
            dict with:
              - "loss": scalar tensor
              - all keys returned by forward() (e.g. "logits", "probs", "preds")
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
        pooling: str = "last",  # 'last', 'mean', 'max', 'mean_max', 'concat'
        mlp_multiplier: float = 2.0,
        use_layer_norm: bool = True,
    ):
        super().__init__(input_dim, num_classes)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pooling = pooling.lower()
        self.use_layer_norm = use_layer_norm

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)

        if self.pooling == "mean_max":
            feature_dim = lstm_out_dim * 2
        elif self.pooling == "concat":
            feature_dim = lstm_out_dim * 2
        else:
            feature_dim = lstm_out_dim

        hidden_fc = max(int(feature_dim * mlp_multiplier), feature_dim)
        layers = []
        if self.use_layer_norm:
            layers.append(nn.LayerNorm(feature_dim))
        layers.extend(
            [
                nn.Linear(feature_dim, hidden_fc),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_fc, num_classes),
            ]
        )
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, T, D)
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (B, T, H*)

        if self.pooling == "mean":
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
            # Default to last time step
            features = lstm_out[:, -1, :]

        logits = self.classifier(features)  # (B, C)
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)

        return {
            "logits": logits,
            "probs": probs,
            "preds": preds,
        }


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
    ):
        super().__init__(input_dim, num_classes)

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=input_dim,
                    out_channels=num_filters,
                    kernel_size=k,
                    padding=k // 2,
                )
                for k in kernel_sizes
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        # x: (B, T, D) -> (B, D, T) for Conv1d
        x = x.transpose(1, 2)  # (B, D, T)

        conv_outs = []
        for conv in self.convs:
            h = conv(x)  # (B, num_filters, T)
            h = F.relu(h)
            # Global max pool over time dimension
            h = F.adaptive_max_pool1d(h, 1).squeeze(-1)  # (B, num_filters)
            conv_outs.append(h)

        features = torch.cat(conv_outs, dim=1)  # (B, num_filters * len(kernel_sizes))
        features = self.dropout(features)

        logits = self.fc(features)  # (B, C)
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)

        return {
            "logits": logits,
            "probs": probs,
            "preds": preds,
        }


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
    ):
        super().__init__(input_dim, num_classes)
        self.pooling = pooling.lower()

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
        if self.pooling == "mean_max":
            pooled_dim = lstm_out_dim * 2
        elif self.pooling == "concat":
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
        # x: (B, T, D)
        h = x.transpose(1, 2)  # (B, D, T)

        # Parallel multi-kernel CNN
        branch_outs = [branch(h) for branch in self.conv_branches]
        h = torch.cat(branch_outs, dim=1)  # (B, concat_filters, T)

        # Project to cnn_out_dim
        h = self.cnn_projection(h)  # (B, cnn_out_dim, T)
        h = h.transpose(1, 2)  # (B, T, cnn_out_dim)

        # BiLSTM
        lstm_out, _ = self.lstm(h)  # (B, T, lstm_out_dim)
        lstm_out = self.lstm_norm(lstm_out)

        # Pooling
        if self.pooling == "mean":
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
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)

        return {
            "logits": logits,
            "probs": probs,
            "preds": preds,
        }


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
    ):
        super().__init__(input_dim, num_classes)

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
        # x: (B, T, D)
        H, _ = self.lstm(x)  # (B, T, 2H)

        # Compute attention scores
        # (B, T, attn_dim)
        attn_scores = torch.tanh(self.attn(H))
        # Project to scalar score per time-step: (B, T)
        attn_scores = torch.matmul(attn_scores, self.attn_vector)
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # (B, T, 1)

        # Weighted sum of hidden states
        context = (H * attn_weights).sum(dim=1)  # (B, 2H)
        context = self.dropout(context)

        logits = self.fc(context)  # (B, C)
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)

        return {
            "logits": logits,
            "probs": probs,
            "preds": preds,
            "attn_weights": attn_weights.squeeze(-1),  # (B, T)
        }


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
        pooling: str = "mean",  # "mean" or "cls"
    ):
        super().__init__(input_dim, num_classes)

        self.d_model = d_model
        self.pooling = pooling

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # so we can keep (B, T, d_model)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if pooling == "cls":
            # learnable classification token
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        else:
            self.cls_token = None

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x, src_key_padding_mask=None):
        """
        Args:
            x: Tensor (B, T, D)
            src_key_padding_mask: optional BoolTensor (B, T) where True means "pad" / ignore.

        Returns:
            dict with "logits", "probs", "preds"
        """
        B, T, D = x.shape
        h = self.input_proj(x)  # (B, T, d_model)

        if self.pooling == "cls":
            # prepend cls token to sequence
            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
            h = torch.cat([cls_tokens, h], dim=1)  # (B, 1+T, d_model)

            if src_key_padding_mask is not None:
                # add padding mask for cls token (never padded)
                cls_pad = torch.zeros(
                    B, 1, dtype=torch.bool, device=src_key_padding_mask.device
                )
                src_key_padding_mask = torch.cat(
                    [cls_pad, src_key_padding_mask], dim=1
                )

        h = self.pos_encoding(h)  # (B, T', d_model)
        # TransformerEncoder expects src_key_padding_mask shape (B, T')
        encoded = self.encoder(h, src_key_padding_mask=src_key_padding_mask)  # (B, T', d_model)

        if self.pooling == "cls":
            # use first token
            features = encoded[:, 0, :]  # (B, d_model)
        else:
            # mean pooling over non-padded positions
            if src_key_padding_mask is not None:
                # src_key_padding_mask: True for pads, False for real
                mask = ~src_key_padding_mask  # True for real tokens
                mask = mask.unsqueeze(-1)  # (B, T, 1)
                encoded = encoded * mask  # zero-out pads
                lengths = mask.sum(dim=1).clamp(min=1)  # (B, 1)
                features = encoded.sum(dim=1) / lengths  # (B, d_model)
            else:
                features = encoded.mean(dim=1)  # (B, d_model)

        features = self.dropout(features)
        logits = self.fc(features)  # (B, C)
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)

        return {
            "logits": logits,
            "probs": probs,
            "preds": preds,
        }

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
    ):
        super().__init__(input_dim, num_classes)
        self.pooling = pooling.lower()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, expand, conv_kernel, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        if self.pooling == "mean_max":
            pooled_dim = d_model * 2
        elif self.pooling == "concat":
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
        # x: (B, T, input_dim)
        h = self.input_proj(x)          # (B, T, d_model)
        for block in self.blocks:
            h = block(h)                 # (B, T, d_model)
        h = self.final_norm(h)

        if self.pooling == "mean":
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
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        return {"logits": logits, "probs": probs, "preds": preds}


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
    ):
        super().__init__(input_dim, num_classes)

        # Build Inception stack
        blocks = []
        in_ch = input_dim
        for i in range(n_inception_blocks):
            block = InceptionBlock(in_ch, num_filters, kernel_sizes, use_residual=True)
            # InceptionBlock output channels
            in_ch = num_filters * len(kernel_sizes) + num_filters
            blocks.append(block)
        self.inception_stack = nn.Sequential(*blocks)
        feat_dim = in_ch  # channels after inception

        # Squeeze-and-Excitation channel attention
        self.se = SqueezeExcitation(feat_dim, reduction=se_reduction)

        # Multi-head self-attention over time
        self.attn_norm = nn.LayerNorm(feat_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=feat_dim, num_heads=n_attn_heads,
            dropout=attn_dropout, batch_first=True,
        )

        # Pooling: concat global avg + global max
        pooled_dim = feat_dim * 2

        # MLP head
        hidden_fc = max(int(pooled_dim * mlp_multiplier), pooled_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hidden_fc),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_fc, num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict:
        # x: (B, T, D)
        h = x.transpose(1, 2)                 # (B, D, T)

        # Multi-scale convolutions
        h = self.inception_stack(h)            # (B, feat_dim, T)

        # Channel attention
        h = self.se(h)                         # (B, feat_dim, T)

        # Self-attention over time dimension
        h = h.transpose(1, 2)                  # (B, T, feat_dim)
        h_norm = self.attn_norm(h)
        h_attn, _ = self.self_attn(h_norm, h_norm, h_norm)
        h = h + h_attn                         # residual

        # Pool: concat(global_avg, global_max)
        h_t = h.transpose(1, 2)               # (B, feat_dim, T)
        avg_pool = F.adaptive_avg_pool1d(h_t, 1).squeeze(-1)  # (B, feat_dim)
        max_pool = F.adaptive_max_pool1d(h_t, 1).squeeze(-1)  # (B, feat_dim)
        features = torch.cat([avg_pool, max_pool], dim=1)      # (B, 2*feat_dim)

        logits = self.classifier(features)
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        return {"logits": logits, "probs": probs, "preds": preds}


class VaeTebTimeSeriesClassifier(nn.Module):
    """
    Combined VAE + Classifier model for time series classification.

    This model uses a pre-trained VAE (SeqVae) to encode time series data into
    latent representations, then classifies those representations using a
    downstream classifier.

    Architecture:
        1. VAE Encoder: Encodes (y_st, y_ph, x_ph) -> z (B, T, D)
        2. Classifier: Maps z -> class logits (B, num_classes)

    Args:
        vae_model: Pre-trained SeqVae model for encoding
        classifier: Any classifier from BaseTimeSeriesClassifier (LSTM, CNN, Transformer, etc.)
        freeze_vae: If True, freeze VAE weights during training (default: True)
        use_posterior: If True, use posterior z from q(z|x,y); else use prior mu from p(z|y)
        sample_latent: If True, sample from latent distribution; else use mean
    """
    def __init__(
        self,
        vae_model: SeqVae,
        classifier: BaseTimeSeriesClassifier,
        freeze_vae: bool = True,
        use_posterior: bool = True,
        sample_latent: bool = False,
        class_weights: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        self.vae_model = vae_model
        self.classifier = classifier
        self.freeze_vae = freeze_vae
        self.use_posterior = use_posterior
        self.sample_latent = sample_latent
        if class_weights is not None:
            weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", weight_tensor)
        else:
            # Keep an attribute for convenience when no weights are provided
            self.class_weights = None

        if self.freeze_vae:
            for param in self.vae_model.parameters():
                param.requires_grad = False
            self.vae_model.eval()

    def encode_features(
        self,
        y_st: torch.Tensor,
        y_ph: torch.Tensor,
        x_ph: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract latent features from VAE encoder.

        Args:
            y_st: Target scattering features (B, T, 43)
            y_ph: Target phase harmonic features (B, T, 44)
            x_ph: Source cross-phase + UP self-phase features (B, T, 137)

        Returns:
            z: Latent features (B, T, D) where D is latent_dim
        """
        if self.freeze_vae:
            self.vae_model.eval()
            with torch.no_grad():
                enc_outputs = self.vae_model.encode_only(
                    y_st=y_st,
                    y_ph=y_ph,
                    x_ph=x_ph,
                    sample_z=self.sample_latent
                )
        else:
            enc_outputs = self.vae_model.encode_only(
                y_st=y_st,
                y_ph=y_ph,
                x_ph=x_ph,
                sample_z=self.sample_latent
            )

        if self.use_posterior:
            if self.sample_latent:
                z = enc_outputs["z"]  # Sampled from q(z|x,y)
            else:
                z = enc_outputs["mu_post"]  # Mean of q(z|x,y)
        else:
            z = enc_outputs["mu_prior"]  # Mean of p(z|y)

        return z  # (B, T, D)

    def forward(
        self,
        y_st: torch.Tensor,
        y_ph: torch.Tensor,
        x_ph: torch.Tensor
    ) -> dict:
        """
        Forward pass through VAE encoder + classifier.

        Args:
            y_st: Target scattering features (B, T, 43)
            y_ph: Target phase harmonic features (B, T, 44)
            x_ph: Source cross-phase + UP self-phase features (B, T, 137)

        Returns:
            Dictionary containing:
                - logits: Class logits (B, num_classes)
                - probs: Class probabilities (B, num_classes)
                - preds: Predicted class indices (B,)
                - latent_features: Latent representations (B, T, D)
        """
        z = self.encode_features(y_st, y_ph, x_ph)  # (B, T, D)

        classifier_outputs = self.classifier(z)  # (B, num_classes)

        classifier_outputs["latent_features"] = z

        return classifier_outputs

    def compute_loss(
        self,
        y_st: torch.Tensor,
        y_ph: torch.Tensor,
        x_ph: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict:
        """
        Compute classification loss.

        Args:
            y_st: Target scattering features (B, T, 43)
            y_ph: Target phase harmonic features (B, T, 44)
            x_ph: Source cross-phase + UP self-phase features (B, T, 137)
            labels: Ground truth class labels (B,)

        Returns:
            Dictionary containing:
                - loss: Cross-entropy loss (scalar)
                - logits: Class logits (B, num_classes)
                - probs: Class probabilities (B, num_classes)
                - preds: Predicted class indices (B,)
                - accuracy: Classification accuracy (scalar)
        """
        outputs = self.forward(y_st, y_ph, x_ph)
        logits = outputs["logits"]

        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, labels, weight=getattr(self, "class_weights", None))

        # Compute accuracy
        preds = outputs["preds"]
        accuracy = (preds == labels).float().mean()

        return {
            "loss": loss,
            "accuracy": accuracy,
            **outputs
        }
