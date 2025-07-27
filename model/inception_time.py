import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from collections import OrderedDict
from typing import Optional
warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn')

class FHRInception(nn.Module):
    """
    Optimized Inception block for FHR signal classification.
    Designed for latent representations from SeqVaeTeb with better
    temporal pattern capture and improved initialization.
    """
    def __init__(self, input_size: int, filters: int, dropout: float = 0.1):
        super(FHRInception, self).__init__()
        
        # Optimized kernel sizes for FHR temporal patterns (sequence_length=300)
        # Smaller kernels for fine-grained patterns, larger for long-term trends
        
        self.bottleneck1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=filters,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        
        # Short-term patterns (4-second windows at 4Hz sampling)
        self.conv_short = nn.Conv1d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=5,  # ~5 time steps
            stride=1,
            padding=2,
            bias=False
        )
        
        # Medium-term patterns (variability patterns)
        self.conv_medium = nn.Conv1d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=15,  # ~15 time steps
            stride=1,
            padding=7,
            bias=False
        )
        
        # Long-term patterns (baseline trends)
        self.conv_long = nn.Conv1d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=40,  # ~40 time steps
            stride=1,
            padding=20,
            bias=False
        )
        
        # Pooling branch for capturing max activations
        self.max_pool = nn.MaxPool1d(
            kernel_size=3,
            stride=1,
            padding=1,
        )
        
        self.bottleneck2 = nn.Conv1d(
            in_channels=input_size,
            out_channels=filters,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        
        # Improved normalization and regularization
        self.batch_norm = nn.BatchNorm1d(num_features=4 * filters)
        self.dropout = nn.Dropout1d(dropout)
        
        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier/He initialization for better gradient flow."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass through FHR-optimized Inception block.
        
        Args:
            x: Input tensor (batch_size, channels, sequence_length)
        Returns:
            Multi-scale feature representation
        """
        # First bottleneck
        x0 = self.bottleneck1(x)
        
        # Multi-scale convolutions
        x1 = self.conv_short(x0)   # Short-term patterns
        x2 = self.conv_medium(x0)  # Medium-term patterns  
        x3 = self.conv_long(x0)    # Long-term patterns
        
        # Pooling branch
        x4 = self.bottleneck2(self.max_pool(x))
        
        # Concatenate all branches
        y = torch.concat([x1, x2, x3, x4], dim=1)
        
        # Normalization, activation, and dropout
        y = self.batch_norm(y)
        y = F.relu(y)
        y = self.dropout(y)
        
        return y


class FHRResidual(nn.Module):
    """
    Improved residual connection for FHR classification.
    Includes proper dimension matching and better normalization.
    """
    def __init__(self, input_size: int, filters: int, dropout: float = 0.1):
        super(FHRResidual, self).__init__()
        
        # Ensure dimension matching for residual connection
        self.bottleneck = nn.Conv1d(
            in_channels=input_size,
            out_channels=4 * filters,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        self.batch_norm = nn.BatchNorm1d(num_features=4 * filters)
        self.dropout = nn.Dropout1d(dropout)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for better gradient flow."""
        nn.init.kaiming_normal_(self.bottleneck.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.batch_norm.weight, 1)
        nn.init.constant_(self.batch_norm.bias, 0)
    
    def forward(self, x, y):
        """
        Forward pass with residual connection.
        
        Args:
            x: Original input (for residual connection)
            y: Current feature map
        Returns:
            Feature map with residual connection
        """
        # Apply bottleneck to input for dimension matching
        residual = self.bottleneck(x)
        residual = self.batch_norm(residual)
        
        # Add residual connection
        y = y + residual
        y = F.relu(y)
        y = self.dropout(y)
        
        return y


class Lambda(torch.nn.Module):
    
    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f
    
    def forward(self, x):
        return self.f(x)


class FHRInceptionTimeClassifier(nn.Module):
    """
    Optimized Inception Time model for FHR signal classification from latent representations.
    
    Designed to work with SeqVaeTeb latent outputs (batch, 300, 32) and classify
    FHR patterns for clinical decision making.
    """
    def __init__(
        self, 
        input_size: int = 32,  # Latent dimension from SeqVaeTeb
        num_classes: int = 2,
        filters: int = 32,
        depth: int = 6,
        dropout: float = 0.2,
        use_attention: bool = True
    ):
        super(FHRInceptionTimeClassifier, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        self.filters = filters
        self.depth = depth
        self.dropout = dropout
        self.use_attention = use_attention
        
        # Input projection to match expected dimensions
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, input_size),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Build inception and residual blocks
        self.inception_blocks = nn.ModuleList()
        self.residual_blocks = nn.ModuleList()
        
        for d in range(depth):
            # Inception block
            block_input_size = input_size if d == 0 else 4 * filters
            self.inception_blocks.append(
                FHRInception(
                    input_size=block_input_size,
                    filters=filters,
                    dropout=dropout
                )
            )
            
            # Residual connection every 3 blocks
            if d % 3 == 2:
                residual_input_size = input_size if d == 2 else 4 * filters
                self.residual_blocks.append(
                    FHRResidual(
                        input_size=residual_input_size,
                        filters=filters,
                        dropout=dropout
                    )
                )
        
        # Attention mechanism for sequence aggregation
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=4 * filters,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(4 * filters)
        
        # Global pooling and classification head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(4 * filters, 2 * filters),
            nn.LayerNorm(2 * filters),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * filters, filters),
            nn.LayerNorm(filters),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(filters, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize all weights properly."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass for FHR classification.
        
        Args:
            x: Latent representations from SeqVaeTeb (batch, sequence_length=300, latent_dim=32)
        Returns:
            Classification logits (batch, num_classes)
        """
        batch_size, seq_len, latent_dim = x.shape
        
        # Input projection and normalization
        x = self.input_projection(x)  # (batch, 300, 32)
        
        # Convert to channel-first for convolutions
        x = x.transpose(1, 2)  # (batch, 32, 300)
        
        # Store original input for residual connections
        residual_inputs = [x]
        residual_idx = 0
        
        # Pass through inception blocks with residual connections
        for d in range(self.depth):
            y = self.inception_blocks[d](x if d == 0 else y)
            
            # Apply residual connection every 3 blocks
            if d % 3 == 2:
                y = self.residual_blocks[residual_idx](residual_inputs[residual_idx], y)
                residual_inputs.append(y)
                residual_idx += 1
                x = y
        
        # Apply attention mechanism if enabled
        if self.use_attention:
            # Convert back to sequence-first for attention
            y_seq = y.transpose(1, 2)  # (batch, 300, 4*filters)
            
            # Self-attention
            attn_out, _ = self.attention(y_seq, y_seq, y_seq)
            y_seq = self.attention_norm(y_seq + attn_out)
            
            # Convert back to channel-first
            y = y_seq.transpose(1, 2)  # (batch, 4*filters, 300)
        
        # Global average pooling
        y = self.global_pool(y)  # (batch, 4*filters, 1)
        y = y.squeeze(-1)  # (batch, 4*filters)
        
        # Classification
        logits = self.classifier(y)  # (batch, num_classes)
        
        return logits
