"""
CNN–LSTM Hybrid Model for lncRNA Sequence Classification
--------------------------------------------------------

This module implements a hybrid deep learning architecture combining:
1. Convolutional Neural Networks (CNNs) for local motif detection
2. Bidirectional LSTM layers for long-range sequence dependencies
3. Handcrafted numerical features (GC%, ORF length, MFE, k-mer patterns, etc.)

Author: SUBASH P
Institution: CCS Haryana Agricultural University
Date: 2025
"""

import torch
import torch.nn as nn


class CNNLSTM_Hybrid(nn.Module):
    """
    Hybrid CNN + BiLSTM architecture that integrates handcrafted biological
    features with sequence-derived representations.

    Parameters
    ----------
    feature_dim : int
        Number of handcrafted numerical features appended after LSTM encoding.
        (Example: GC content, ORF length, k-mer counts, etc.)

    input_channels : int
        Number of channels for the input one-hot sequence. Default = 4 (A/C/G/T).

    max_len : int
        Length of the padded input sequence. Default = 300.

    Output
    ------
    logits : torch.Tensor
        Shape: [batch_size, 2]
        Raw prediction scores for binary classification.
    """

    def __init__(self, feature_dim=5, input_channels=4, max_len=300):
        super(CNNLSTM_Hybrid, self).__init__()

        self.feature_dim = feature_dim
        self.input_channels = input_channels
        self.max_len = max_len

        # ---------------------------------------------------------------------
        # CNN MODULE
        # Extract short-range motifs and local sequence patterns
        # Input shape to CNN: [batch_size, 4, max_len]
        # ---------------------------------------------------------------------
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4)
        )

        # ---------------------------------------------------------------------
        # BiLSTM MODULE
        # Learns long-range dependencies after CNN compresses the sequence
        # LSTM input size = number of channels after CNN = 128
        # ---------------------------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # ---------------------------------------------------------------------
        # FULLY CONNECTED CLASSIFIER
        # Combines:
        #  - LSTM latent vector (128 dimensions from BiLSTM)
        #  - Handcrafted features (feature_dim)
        # ---------------------------------------------------------------------
        self.fc = nn.Sequential(
            nn.Linear(128 + feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(64, 2)
        )

    # -------------------------------------------------------------------------
    # FORWARD PASS
    # seq      : One-hot encoded sequence tensor → [batch, max_len, 4]
    # features : Handcrafted numerical features → [batch, feature_dim]
    # -------------------------------------------------------------------------
    def forward(self, seq, features):

        # Convert from [batch, max_len, channels] → [batch, channels, max_len]
        x = seq.permute(0, 2, 1)

        # CNN feature extraction
        x = self.cnn(x)  # → [batch, 128, reduced_length]

        # Convert to LSTM format: [batch, seq_len, features]
        x = x.permute(0, 2, 1)

        # LSTM output: h_n contains final hidden states from all layers & directions
        _, (h_n, _) = self.lstm(x)

        # Concatenate last forward and backward hidden states (BiLSTM)
        lstm_out = torch.cat((h_n[-2], h_n[-1]), dim=1)  # → [batch, 128]

        # Append handcrafted biological features
        combined = torch.cat((lstm_out, features), dim=1)

        # Final classification layer
        logits = self.fc(combined)

        return logits
