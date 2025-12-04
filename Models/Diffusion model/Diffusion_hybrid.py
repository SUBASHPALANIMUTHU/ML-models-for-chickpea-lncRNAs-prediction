"""
Hybrid CNN–Diffusion–LSTM Model for lncRNA Classification
---------------------------------------------------------

This module implements a scientific deep-learning architecture designed for
lncRNA versus mRNA sequence classification. The model integrates:

    • Convolutional Neural Networks (CNNs)
        – capture short-range nucleotide motifs
    • Diffusion-style residual blocks
        – stabilize feature refinement through residual noise-injection dynamics
    • Unidirectional LSTM encoder
        – models long-range contextual dependencies in sequence representations
    • Handcrafted biological features
        – GC content, ORF length score etc.,
Author: SUBASH P
Institution: CCS Haryana Agricultural University
Date: 2025
"""

import torch
import torch.nn as nn


# ============================================================================
# Diffusion Residual Block
# ============================================================================
class DiffusionBlock(nn.Module):
    """
    Diffusion-inspired residual block used to refine local convolutional
    representations. The operation follows a residual update scheme:

        y = ReLU( Conv1d(x) + x )

    Parameters
    ----------
    channels : int
        Number of input and output feature channels.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x) + x)


# ============================================================================
# Hybrid CNN + Diffusion + LSTM Model
# ============================================================================
class HybridDiffusionLSTM(nn.Module):
    """
    Hybrid deep-learning model combining:
        1. CNN convolutional blocks
        2. Diffusion residual refinement
        3. Unidirectional LSTM sequence encoder
        4. Biological feature projection
        5. Final classification head

    Input Shapes
    ------------
    seq_onehot : torch.Tensor
        One-hot encoded nucleotide sequence tensor of shape:
            [batch_size, 4, sequence_length]

    features : torch.Tensor
        Handcrafted feature tensor of shape:
            [batch_size, feature_dim]

    Output
    ------
    logits : torch.Tensor
        Raw unnormalized classification scores of shape:
            [batch_size, 2]
    """

    def __init__(self, feature_dim: int = 4):
        super().__init__()

        # ----------------------------------------------------------------------
        # Convolution + Diffusion Feature Extractor
        # ----------------------------------------------------------------------
        self.cnn = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            DiffusionBlock(64),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            DiffusionBlock(128),

            nn.MaxPool1d(kernel_size=2)
        )

        # ----------------------------------------------------------------------
        # LSTM Encoder (Unidirectional)
        # ----------------------------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            batch_first=True,
            bidirectional=False  # IMPORTANT: LSTM only (no BiLSTM)
        )

        # ----------------------------------------------------------------------
        # Handcrafted Feature Encoder
        # ----------------------------------------------------------------------
        self.feature_fc = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU()
        )

        # ----------------------------------------------------------------------
        # Final Classification Head
        # Combines:
        #    • LSTM representation (64 units)
        #    • Projected biological feature vector (32 units)
        # ----------------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.40),
            nn.Linear(64, 2)
        )

    # =========================================================================
    # Forward Pass
    # =========================================================================
    def forward(self, seq_onehot: torch.Tensor, features: torch.Tensor):
        """
        Parameters
        ----------
        seq_onehot : torch.Tensor
            One-hot encoded sequence (batch × 4 × L)

        features : torch.Tensor
            Auxiliary biological features (batch × feature_dim)

        Returns
        -------
        torch.Tensor
            Unnormalized logits for the two output classes.
        """

        # ----- CNN + Diffusion -----
        x = self.cnn(seq_onehot)                 # [B, 128, L']
        x = x.permute(0, 2, 1)                   # [B, L', 128]

        # ----- LSTM -----
        _, (h_n, _) = self.lstm(x)               # h_n: [1, B, 64]
        seq_repr = h_n[-1]                       # [B, 64]

        # ----- Biological Features -----
        feat_repr = self.feature_fc(features)    # [B, 32]

        # ----- Fusion -----
        merged = torch.cat([seq_repr, feat_repr], dim=1)

        # ----- Classification -----
        logits = self.classifier(merged)
        return logits
