"""
PyTorch Lightning Recurrent Classifier for Light Curve Classification.

This module provides a drop-in replacement for CatBoostClassifier with support for:
- FEATURES_ONLY: Pure MLP on handcrafted features (exact CatBoost replacement)
- SEQUENCE_ONLY: RNN (LSTM/GRU/RNN) on raw time series
- HYBRID: Both sequence and features combined

Example:
    >>> from recurrent_classifier import RecurrentClassifierWrapper, RecurrentConfig, InputMode
    >>> config = RecurrentConfig(input_mode=InputMode.FEATURES_ONLY)
    >>> wrapper = RecurrentClassifierWrapper(config)
    >>> wrapper.fit(features=X_train, labels=y_train)
    >>> proba = wrapper.predict_proba(features=X_test)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    import polars as pl_df

__all__ = [
    "RNNType",
    "InputMode",
    "RecurrentConfig",
    "LightCurveDataset",
    "collate_fn",
    "AttentionPooling",
    "MLPHead",
    "RecurrentLightCurveClassifier",
    "RecurrentClassifierWrapper",
    "extract_sequences",
    "extract_sequences_chunked",
]


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────


class RNNType(str, Enum):
    """Supported sequence encoder architectures."""

    LSTM = "lstm"
    GRU = "gru"
    RNN = "rnn"
    TRANSFORMER = "transformer"


class InputMode(str, Enum):
    """Input data modes for the classifier."""

    SEQUENCE_ONLY = "sequence"  # Raw time series only
    FEATURES_ONLY = "features"  # Handcrafted features only (CatBoost drop-in)
    HYBRID = "hybrid"  # Both sequence + features


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RecurrentConfig:
    """
    Configuration for recurrent classifier.

    Attributes:
        input_mode: Which inputs to use (SEQUENCE_ONLY, FEATURES_ONLY, HYBRID)
        rnn_type: RNN architecture (LSTM, GRU, RNN) - ignored if FEATURES_ONLY
        hidden_size: RNN hidden state size
        num_layers: Number of RNN layers
        bidirectional: Whether to use bidirectional RNN
        use_attention: Whether to use attention pooling (vs last hidden)
        mlp_hidden_sizes: Tuple of MLP hidden layer sizes
        dropout: Dropout probability
        learning_rate: Learning rate for optimizer
        weight_decay: L2 regularization weight
        batch_size: Training batch size
        max_epochs: Maximum training epochs
        early_stopping_patience: Epochs to wait before early stopping
        gradient_clip_val: Gradient clipping value
        accelerator: Device to use ("auto", "gpu", "cpu")
        num_workers: DataLoader workers
    """

    # Input mode
    input_mode: InputMode = InputMode.HYBRID

    # Sequence Encoder Architecture (ignored if input_mode=FEATURES_ONLY)
    rnn_type: RNNType = RNNType.LSTM
    hidden_size: int = 128
    num_layers: int = 2
    bidirectional: bool = True  # For RNN/LSTM/GRU only
    use_attention: bool = True  # For RNN/LSTM/GRU only

    # Transformer-specific (only used if rnn_type=TRANSFORMER)
    n_heads: int = 4  # Number of attention heads
    dim_feedforward: int = 256  # Feedforward dimension in transformer

    # MLP Head
    mlp_hidden_sizes: tuple[int, ...] = (256, 128)
    dropout: float = 0.3

    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 100
    early_stopping_patience: int = 10
    gradient_clip_val: float = 1.0

    # Hardware
    accelerator: str = "auto"
    num_workers: int = 0  # 0 for Windows compatibility

    # Preprocessing
    scale_features: bool = True  # StandardScaler on aux features

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.dropout < 0 or self.dropout >= 1:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────


class LightCurveDataset(Dataset):
    """
    Dataset for light curve sequences with optional auxiliary features.

    Handles variable-length sequences and optional tabular features.
    """

    __slots__ = ("sequences", "aux_features", "labels", "sample_weights", "_has_sequences")

    def __init__(
        self,
        sequences: list[np.ndarray] | None,
        aux_features: np.ndarray | None,
        labels: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> None:
        """
        Initialize dataset.

        Args:
            sequences: List of (seq_len, n_features) arrays, or None
            aux_features: (n_samples, n_features) array, or None
            labels: (n_samples,) array of binary labels
            sample_weights: (n_samples,) array of weights, or None
        """
        self.sequences = sequences
        self.aux_features = aux_features
        self.labels = labels
        self.sample_weights = sample_weights
        self._has_sequences = sequences is not None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single sample."""
        item: dict[str, torch.Tensor] = {
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

        if self._has_sequences:
            item["sequence"] = torch.tensor(self.sequences[idx], dtype=torch.float32)

        if self.aux_features is not None:
            item["aux_features"] = torch.tensor(self.aux_features[idx], dtype=torch.float32)

        if self.sample_weights is not None:
            item["sample_weights"] = torch.tensor(self.sample_weights[idx], dtype=torch.float32)

        return item


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    Collate function handling variable-length sequences.

    Pads sequences to max length in batch.

    Args:
        batch: List of sample dicts from dataset

    Returns:
        Collated batch dict with:
        - sequences: (batch, max_len, n_features) padded tensor
        - lengths: (batch,) original lengths
        - aux_features: (batch, n_features) if present
        - labels: (batch,)
        - sample_weights: (batch,) if present
    """
    result: dict[str, torch.Tensor] = {}

    # Labels (always present)
    result["labels"] = torch.stack([item["labels"] for item in batch])

    # Sequences (variable length)
    if "sequence" in batch[0]:
        sequences = [item["sequence"] for item in batch]
        lengths = torch.tensor([seq.size(0) for seq in sequences], dtype=torch.long)
        padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
        result["sequences"] = padded
        result["lengths"] = lengths

    # Auxiliary features
    if "aux_features" in batch[0]:
        result["aux_features"] = torch.stack([item["aux_features"] for item in batch])

    # Sample weights
    if "sample_weights" in batch[0]:
        result["sample_weights"] = torch.stack([item["sample_weights"] for item in batch])

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Model Components
# ─────────────────────────────────────────────────────────────────────────────


class AttentionPooling(nn.Module):
    """
    Attention mechanism to aggregate variable-length RNN outputs.

    Learns importance weights for each timestep, producing a fixed-size
    context vector regardless of sequence length.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        rnn_output: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply attention pooling.

        Args:
            rnn_output: (batch, seq_len, hidden) RNN outputs
            lengths: (batch,) original sequence lengths

        Returns:
            Context vector (batch, hidden)
        """
        batch_size, max_len, hidden_size = rnn_output.size()
        device = rnn_output.device

        # Compute attention scores
        scores = self.attention(rnn_output).squeeze(-1)  # (batch, seq_len)

        # Create mask for padded positions
        mask = torch.arange(max_len, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
        scores = scores.masked_fill(mask, float("-inf"))

        # Softmax over valid positions
        attention_weights = torch.softmax(scores, dim=1)  # (batch, seq_len)

        # Weighted sum
        context = torch.bmm(attention_weights.unsqueeze(1), rnn_output).squeeze(1)

        return context


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer.

    Adds position information to input embeddings since Transformers
    have no inherent notion of sequence order.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])

        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input. x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerSequenceEncoder(nn.Module):
    """
    Transformer encoder for variable-length sequences.

    Projects input to hidden_size, applies positional encoding,
    then runs through TransformerEncoder layers.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        n_heads: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Project input to hidden_size
        self.input_projection = nn.Linear(input_size, hidden_size)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_size, dropout=dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # CLS token for classification (learnable)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))

        self.hidden_size = hidden_size

    def forward(
        self,
        sequences: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode sequences with Transformer.

        Args:
            sequences: (batch, seq_len, input_size) padded sequences
            lengths: (batch,) original sequence lengths

        Returns:
            Context vector (batch, hidden_size) from CLS token
        """
        batch_size, max_len, _ = sequences.size()
        device = sequences.device

        # Project to hidden size
        x = self.input_projection(sequences)  # (batch, seq_len, hidden_size)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, hidden_size)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, seq_len+1, hidden_size)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Create attention mask for padded positions (True = ignore)
        # Account for CLS token at position 0
        seq_positions = torch.arange(max_len + 1, device=device).unsqueeze(0)  # (1, seq_len+1)
        # CLS token (position 0) is always valid, so we compare with lengths+1
        padding_mask = seq_positions > lengths.unsqueeze(1)  # (batch, seq_len+1)

        # Run through transformer
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # Return CLS token representation
        return x[:, 0, :]  # (batch, hidden_size)


class MLPHead(nn.Module):
    """
    MLP classification head.

    Used by all input modes for final classification.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: tuple[int, ...],
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_size = input_size

        layers: list[nn.Module] = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        return self.mlp(x)


# ─────────────────────────────────────────────────────────────────────────────
# Lightning Module
# ─────────────────────────────────────────────────────────────────────────────


class RecurrentLightCurveClassifier(pl.LightningModule):
    """
    PyTorch Lightning module for light curve classification.

    Supports three input modes:
    - SEQUENCE_ONLY: RNN on raw time series
    - FEATURES_ONLY: MLP on handcrafted features (CatBoost replacement)
    - HYBRID: Both combined
    """

    def __init__(
        self,
        config: RecurrentConfig,
        seq_input_size: int = 4,
        aux_input_size: int = 0,
        class_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["class_weight"])
        self.config = config
        self.class_weight = class_weight

        # Build components based on input mode
        self._build_model(seq_input_size, aux_input_size)
        self._setup_metrics()

    def _build_model(self, seq_input_size: int, aux_input_size: int) -> None:
        """Construct model components based on input mode."""
        mlp_input_size = 0
        self._use_transformer = False

        if self.config.input_mode != InputMode.FEATURES_ONLY:
            if self.config.rnn_type == RNNType.TRANSFORMER:
                # Build Transformer encoder
                self._use_transformer = True
                self.transformer_encoder = TransformerSequenceEncoder(
                    input_size=seq_input_size,
                    hidden_size=self.config.hidden_size,
                    num_layers=self.config.num_layers,
                    n_heads=self.config.n_heads,
                    dim_feedforward=self.config.dim_feedforward,
                    dropout=self.config.dropout,
                )
                mlp_input_size += self.config.hidden_size
            else:
                # Build RNN (LSTM/GRU/RNN)
                rnn_class = {
                    RNNType.LSTM: nn.LSTM,
                    RNNType.GRU: nn.GRU,
                    RNNType.RNN: nn.RNN,
                }[self.config.rnn_type]

                self.rnn = rnn_class(
                    input_size=seq_input_size,
                    hidden_size=self.config.hidden_size,
                    num_layers=self.config.num_layers,
                    batch_first=True,
                    bidirectional=self.config.bidirectional,
                    dropout=self.config.dropout if self.config.num_layers > 1 else 0,
                )

                rnn_output_size = self.config.hidden_size * (2 if self.config.bidirectional else 1)

                if self.config.use_attention:
                    self.attention = AttentionPooling(rnn_output_size)

                mlp_input_size += rnn_output_size

        if self.config.input_mode != InputMode.SEQUENCE_ONLY:
            mlp_input_size += aux_input_size

        # Build MLP head
        self.mlp_head = MLPHead(
            input_size=mlp_input_size,
            hidden_sizes=self.config.mlp_hidden_sizes,
            num_classes=2,
            dropout=self.config.dropout,
        )

    def _setup_metrics(self) -> None:
        """Initialize torchmetrics for proper metric aggregation."""
        try:
            from torchmetrics import AUROC, Accuracy, AveragePrecision

            self.train_acc = Accuracy(task="binary")
            self.val_acc = Accuracy(task="binary")
            self.val_auroc = AUROC(task="binary")
            self.val_auprc = AveragePrecision(task="binary")
            self._has_metrics = True
        except ImportError:
            self._has_metrics = False
            warnings.warn("torchmetrics not installed, skipping metric logging")

    def forward(
        self,
        sequences: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
        aux_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            sequences: (batch, seq_len, 4) padded sequences
            lengths: (batch,) original sequence lengths
            aux_features: (batch, n_features) handcrafted features

        Returns:
            Logits (batch, 2)
        """
        features_list: list[torch.Tensor] = []

        # Process sequences if needed
        if self.config.input_mode != InputMode.FEATURES_ONLY:
            if sequences is None or lengths is None:
                raise ValueError("sequences and lengths required for this input mode")
            rnn_out = self._encode_sequences(sequences, lengths)
            features_list.append(rnn_out)

        # Add auxiliary features if needed
        if self.config.input_mode != InputMode.SEQUENCE_ONLY:
            if aux_features is None:
                raise ValueError("aux_features required for this input mode")
            features_list.append(aux_features)

        # Concatenate and classify
        if len(features_list) > 1:
            combined = torch.cat(features_list, dim=1)
        else:
            combined = features_list[0]

        return self.mlp_head(combined)

    def _encode_sequences(
        self,
        sequences: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Encode sequences through RNN or Transformer."""
        if self._use_transformer:
            # Use Transformer encoder (handles its own positional encoding + CLS token)
            return self.transformer_encoder(sequences, lengths)

        # RNN path: Pack for efficient processing
        packed = pack_padded_sequence(
            sequences,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.rnn(packed)
        rnn_out, _ = pad_packed_sequence(packed_out, batch_first=True)

        # Pool to fixed size
        if self.config.use_attention:
            return self.attention(rnn_out, lengths)
        else:
            return self._get_last_hidden(rnn_out, lengths)

    @staticmethod
    def _get_last_hidden(rnn_out: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Extract last valid hidden state for each sequence."""
        device = rnn_out.device
        batch_size = rnn_out.size(0)
        hidden_size = rnn_out.size(2)

        # Get indices for last valid timestep
        last_indices = (lengths - 1).to(device).view(-1, 1, 1).expand(-1, 1, hidden_size)
        return rnn_out.gather(1, last_indices).squeeze(1)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Training step with sample-weighted loss."""
        logits = self._forward_batch(batch)
        loss = self._compute_weighted_loss(
            logits,
            batch["labels"],
            batch.get("sample_weights"),
        )
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        if self._has_metrics:
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs >= 0.5).long()
            self.train_acc(preds, batch["labels"])
            self.log("train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        """Validation step with metrics logging."""
        logits = self._forward_batch(batch)
        loss = self._compute_weighted_loss(logits, batch["labels"], None)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        if self._has_metrics:
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs >= 0.5).long()
            self.val_acc(preds, batch["labels"])
            self.val_auroc(probs, batch["labels"])
            self.val_auprc(probs, batch["labels"])
            self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
            self.log("val_auroc", self.val_auroc, prog_bar=True, on_step=False, on_epoch=True)
            self.log("val_auprc", self.val_auprc, on_step=False, on_epoch=True)

    def _forward_batch(self, batch: dict) -> torch.Tensor:
        """Helper to forward a batch dict."""
        return self(
            sequences=batch.get("sequences"),
            lengths=batch.get("lengths"),
            aux_features=batch.get("aux_features"),
        )

    def _compute_weighted_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute cross-entropy with optional sample weights."""
        if self.class_weight is not None:
            class_weight = self.class_weight.to(logits.device)
        else:
            class_weight = None

        if sample_weights is not None:
            loss_fn = nn.CrossEntropyLoss(weight=class_weight, reduction="none")
            losses = loss_fn(logits, labels)
            return (losses * sample_weights).mean()
        else:
            loss_fn = nn.CrossEntropyLoss(weight=class_weight)
            return loss_fn(logits, labels)

    def configure_optimizers(self):
        """Configure AdamW with OneCycleLR scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Use OneCycleLR if trainer is available
        if self.trainer and self.trainer.estimated_stepping_batches:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.config.learning_rate,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.1,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }

        return optimizer


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper (Sklearn/CatBoost-compatible API)
# ─────────────────────────────────────────────────────────────────────────────


class RecurrentClassifierWrapper:
    """
    Sklearn/CatBoost-compatible wrapper for RecurrentLightCurveClassifier.

    Provides fit/predict/predict_proba interface matching CatBoostClassifier.
    """

    def __init__(
        self,
        config: RecurrentConfig | None = None,
        random_state: int = 42,
    ) -> None:
        self.config = config or RecurrentConfig()
        self.random_state = random_state
        self.model: RecurrentLightCurveClassifier | None = None
        self.trainer: pl.Trainer | None = None
        self._aux_input_size: int = 0
        self._seq_input_size: int = 4

        # Feature scaler (fitted during training)
        self._feature_scaler: StandardScaler | None = None

        # Cache for efficient repeated predictions
        self._prediction_cache: dict[int, np.ndarray] = {}

    def fit(
        self,
        features: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        sequences: list[np.ndarray] | None = None,
        eval_set: tuple | None = None,
        class_weight: dict[int, float] | None = None,
        plot: bool = False,
        plot_file: str | Path | None = None,
    ) -> "RecurrentClassifierWrapper":
        """
        Train the model.

        Args:
            features: (n_samples, n_features) tabular features
            labels: (n_samples,) binary labels
            sample_weight: (n_samples,) per-sample weights
            sequences: List of (seq_len, 4) arrays
            eval_set: Validation data tuple
            class_weight: Class weights dict {0: w0, 1: w1}
            plot: Whether to enable logging
            plot_file: Path for logs (unused, for compatibility)

        Returns:
            self for method chaining
        """
        if labels is None:
            raise ValueError("labels is required")

        self._validate_inputs(features, sequences)
        self._clear_cache()

        # Fit feature scaler on training data
        if self.config.scale_features and features is not None:
            self._feature_scaler = StandardScaler()
            self._feature_scaler.fit(features)

        # Set random seed
        pl.seed_everything(self.random_state, workers=True)

        # Prepare datasets
        train_dataset = self._create_dataset(sequences, features, labels, sample_weight)
        val_dataset = self._create_eval_dataset(eval_set) if eval_set else None

        # Prepare data loaders
        train_loader = self._create_dataloader(train_dataset, shuffle=True)
        val_loader = self._create_dataloader(val_dataset, shuffle=False) if val_dataset else None

        # Store sizes for save/load
        self._aux_input_size = features.shape[1] if features is not None else 0
        self._seq_input_size = sequences[0].shape[1] if sequences is not None and len(sequences) > 0 else 4

        # Initialize model
        self.model = self._create_model(
            seq_input_size=self._seq_input_size,
            aux_input_size=self._aux_input_size,
            class_weight=class_weight,
        )

        # Configure trainer
        self.trainer = self._create_trainer(val_loader is not None, plot)

        # Train
        self.trainer.fit(self.model, train_loader, val_loader)

        return self

    def predict_proba(
        self,
        features: np.ndarray | None = None,
        sequences: list[np.ndarray] | None = None,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            features: (n_samples, n_features) tabular features
            sequences: List of (seq_len, 4) arrays
            batch_size: Override batch size for prediction

        Returns:
            (n_samples, 2) array of probabilities
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        self._validate_inputs(features, sequences)

        # Check cache
        cache_key = self._compute_cache_key(features, sequences)
        if cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key]

        # Determine number of samples
        n_samples = len(sequences) if sequences is not None else len(features)

        # Prepare dataset
        dataset = self._create_dataset(
            sequences,
            features,
            labels=np.zeros(n_samples, dtype=np.int64),
        )
        loader = self._create_dataloader(dataset, shuffle=False, batch_size=batch_size)

        # Predict
        self.model.eval()
        device = next(self.model.parameters()).device
        all_probs: list[np.ndarray] = []

        with torch.no_grad():
            for batch in loader:
                batch = self._batch_to_device(batch, device)
                logits = self.model(
                    sequences=batch.get("sequences"),
                    lengths=batch.get("lengths"),
                    aux_features=batch.get("aux_features"),
                )
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)

        result = np.concatenate(all_probs, axis=0).astype(np.float32)

        # Cache result
        self._prediction_cache[cache_key] = result
        return result

    def predict(
        self,
        features: np.ndarray | None = None,
        sequences: list[np.ndarray] | None = None,
    ) -> np.ndarray:
        """
        Predict class labels.

        Args:
            features: (n_samples, n_features) tabular features
            sequences: List of (seq_len, 4) arrays

        Returns:
            (n_samples,) array of predictions (0 or 1)
        """
        proba = self.predict_proba(features, sequences)
        return (proba[:, 1] >= 0.5).astype(np.int8)

    def _validate_inputs(
        self,
        features: np.ndarray | None,
        sequences: list[np.ndarray] | None,
    ) -> None:
        """Validate inputs match the configured input mode."""
        mode = self.config.input_mode

        if mode == InputMode.FEATURES_ONLY and features is None:
            raise ValueError("features required for FEATURES_ONLY mode")
        if mode == InputMode.SEQUENCE_ONLY and sequences is None:
            raise ValueError("sequences required for SEQUENCE_ONLY mode")
        if mode == InputMode.HYBRID and (features is None or sequences is None):
            raise ValueError("both features and sequences required for HYBRID mode")

    def _create_dataset(
        self,
        sequences: list[np.ndarray] | None,
        features: np.ndarray | None,
        labels: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> LightCurveDataset:
        """Create dataset with proper preprocessing."""
        # Preprocess sequences: compute delta MJD
        processed_seqs = None
        if sequences is not None:
            processed_seqs = [self._preprocess_sequence(seq) for seq in sequences]

        # Scale features if scaler is fitted
        scaled_features = features
        if features is not None and self._feature_scaler is not None:
            scaled_features = self._feature_scaler.transform(features).astype(np.float32)

        return LightCurveDataset(
            sequences=processed_seqs,
            aux_features=scaled_features,
            labels=labels,
            sample_weights=sample_weights,
        )

    def _create_eval_dataset(self, eval_set: tuple) -> LightCurveDataset:
        """Create validation dataset from eval_set tuple."""
        # Handle different tuple formats
        if len(eval_set) == 2:
            # (features, labels) for FEATURES_ONLY
            features, labels = eval_set
            return self._create_dataset(None, features, labels)
        elif len(eval_set) == 3:
            # (sequences, features, labels) for HYBRID
            sequences, features, labels = eval_set
            return self._create_dataset(sequences, features, labels)
        else:
            raise ValueError(f"eval_set must have 2 or 3 elements, got {len(eval_set)}")

    @staticmethod
    def _preprocess_sequence(seq: np.ndarray) -> np.ndarray:
        """
        Preprocess a single sequence with proper normalization.

        For each column:
        - Column 0 (mjd): Delta encode (time differences), then scale by 1/10
        - Column 1 (mag): Z-score normalize (subtract mean, divide by std)
        - Column 2+ (magerr, etc.): Z-score normalize

        TODO: Make preprocessing configurable by column name (e.g., preprocess_config dict)
        instead of hardcoded by position. Would allow specifying "delta_scale", "zscore",
        "none", etc. per column for derived cols like norm/vel.
        """
        result = seq.copy().astype(np.float32)
        n_cols = result.shape[1]

        # Column 0: Delta encode MJD and scale
        if n_cols > 0:
            delta_mjd = np.zeros(len(result), dtype=np.float32)
            delta_mjd[1:] = np.diff(seq[:, 0])
            # Scale delta time (typical gaps are 0.01-10 days, scale to ~[-1, 1])
            result[:, 0] = delta_mjd / 10.0

        # Column 1: Z-score normalize magnitude
        if n_cols > 1:
            mag = seq[:, 1]
            mag_mean = np.mean(mag)
            mag_std = np.std(mag)
            if mag_std > 1e-8:
                result[:, 1] = (mag - mag_mean) / mag_std
            else:
                result[:, 1] = 0.0

        # Columns 2+: Z-score normalize (magerr, etc.)
        for col_idx in range(2, n_cols):
            col = seq[:, col_idx]
            col_mean = np.mean(col)
            col_std = np.std(col)
            if col_std > 1e-8:
                result[:, col_idx] = (col - col_mean) / col_std
            else:
                result[:, col_idx] = 0.0

        return result

    def _create_dataloader(
        self,
        dataset: LightCurveDataset,
        shuffle: bool,
        batch_size: int | None = None,
    ) -> DataLoader:
        """Create DataLoader with proper collate function."""
        return DataLoader(
            dataset,
            batch_size=batch_size or self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.config.num_workers > 0,
        )

    def _create_model(
        self,
        seq_input_size: int,
        aux_input_size: int,
        class_weight: dict[int, float] | None,
    ) -> RecurrentLightCurveClassifier:
        """Create model instance."""
        weight_tensor = None
        if class_weight:
            weight_tensor = torch.tensor(
                [class_weight.get(0, 1.0), class_weight.get(1, 1.0)],
                dtype=torch.float32,
            )

        return RecurrentLightCurveClassifier(
            config=self.config,
            seq_input_size=seq_input_size,
            aux_input_size=aux_input_size,
            class_weight=weight_tensor,
        )

    def _create_trainer(self, has_validation: bool, plot: bool) -> pl.Trainer:
        """Create Lightning Trainer with callbacks."""
        callbacks: list = []

        if has_validation:
            callbacks.append(
                pl.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.config.early_stopping_patience,
                    mode="min",
                )
            )

        return pl.Trainer(
            max_epochs=self.config.max_epochs,
            accelerator=self.config.accelerator,
            callbacks=callbacks,
            gradient_clip_val=self.config.gradient_clip_val,
            enable_progress_bar=True,
            enable_model_summary=False,
            logger=plot,
            deterministic="warn",  # "warn" instead of True to avoid CUDA CuBLAS errors
        )

    def _clear_cache(self) -> None:
        """Clear prediction cache."""
        self._prediction_cache.clear()

    @staticmethod
    def _compute_cache_key(
        features: np.ndarray | None,
        sequences: list[np.ndarray] | None,
    ) -> int:
        """Compute cache key from input arrays."""
        parts: list[int] = []
        if features is not None:
            parts.append(hash(features.tobytes()))
        if sequences is not None:
            parts.append(hash(tuple(s.tobytes() for s in sequences)))
        return hash(tuple(parts))

    @staticmethod
    def _batch_to_device(batch: dict, device: torch.device) -> dict:
        """Move batch tensors to device."""
        return {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    # ─────────────────────────────────────────────────────────────────────
    # Serialization
    # ─────────────────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """
        Save model to disk.

        Args:
            path: File path (recommended: .pt extension)
        """
        if self.model is None:
            raise RuntimeError("No model to save. Call fit() first.")

        state = {
            "config": self.config,
            "model_state_dict": self.model.state_dict(),
            "random_state": self.random_state,
            "aux_input_size": self._aux_input_size,
            "seq_input_size": self._seq_input_size,
            "feature_scaler": self._feature_scaler,
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str | Path) -> "RecurrentClassifierWrapper":
        """
        Load model from disk.

        Args:
            path: File path to saved model

        Returns:
            Loaded wrapper instance
        """
        state = torch.load(path, map_location="cpu", weights_only=False)

        wrapper = cls(config=state["config"], random_state=state["random_state"])
        wrapper._aux_input_size = state.get("aux_input_size", 0)
        wrapper._seq_input_size = state.get("seq_input_size", 4)
        wrapper._feature_scaler = state.get("feature_scaler", None)

        # Reconstruct model
        wrapper.model = RecurrentLightCurveClassifier(
            config=state["config"],
            seq_input_size=wrapper._seq_input_size,
            aux_input_size=wrapper._aux_input_size,
        )
        wrapper.model.load_state_dict(state["model_state_dict"])
        wrapper.model.eval()

        return wrapper


# ─────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────────────────────


def extract_sequences(
    df: "pl_df.DataFrame",
    indices: np.ndarray | list[int] | None = None,
    columns: tuple[str, ...] = ("mjd", "mag", "magerr", "norm"),
) -> list[np.ndarray]:
    """
    Extract raw time series from Polars DataFrame with list columns.

    Args:
        df: DataFrame with list columns (mjd, mag, magerr, norm)
        indices: Optional subset of row indices to extract
        columns: Column names to stack into sequences

    Returns:
        List of (seq_len, n_columns) float32 arrays
    """
    if indices is not None:
        df = df[indices]

    # Extract all columns as Python lists
    col_data = [df[col].to_list() for col in columns]

    # Stack into arrays
    return [
        np.column_stack(row_values).astype(np.float32)
        for row_values in zip(*col_data)
    ]


def extract_sequences_chunked(
    df: "pl_df.DataFrame",
    indices: np.ndarray | list[int] | None = None,
    chunk_size: int = 100_000,
    columns: tuple[str, ...] = ("mjd", "mag", "magerr", "norm"),
) -> list[np.ndarray]:
    """
    Memory-efficient sequence extraction for large datasets.

    Args:
        df: DataFrame with list columns
        indices: Optional subset of row indices
        chunk_size: Number of rows per chunk
        columns: Column names to extract

    Returns:
        List of (seq_len, n_columns) float32 arrays
    """
    if indices is not None:
        indices = np.asarray(indices)
    else:
        indices = np.arange(len(df))

    sequences: list[np.ndarray] = []

    for start in range(0, len(indices), chunk_size):
        chunk_indices = indices[start : start + chunk_size]
        chunk_seqs = extract_sequences(df, chunk_indices.tolist(), columns)
        sequences.extend(chunk_seqs)

    return sequences
