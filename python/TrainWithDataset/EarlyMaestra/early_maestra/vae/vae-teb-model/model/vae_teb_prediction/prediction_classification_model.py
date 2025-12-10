from vae_teb_model_prediction import *

import torch
import torch.nn as nn
import torch.nn.functional as F


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
# 3. BiLSTM + self-attention classifier
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
    ):
        super().__init__()
        self.vae_model = vae_model
        self.classifier = classifier
        self.freeze_vae = freeze_vae
        self.use_posterior = use_posterior
        self.sample_latent = sample_latent

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
            x_ph: Source cross-phase features (B, T, 130)

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
            x_ph: Source cross-phase features (B, T, 130)

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
            x_ph: Source cross-phase features (B, T, 130)
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
        loss = F.cross_entropy(logits, labels)

        # Compute accuracy
        preds = outputs["preds"]
        accuracy = (preds == labels).float().mean()

        return {
            "loss": loss,
            "accuracy": accuracy,
            **outputs
        }
