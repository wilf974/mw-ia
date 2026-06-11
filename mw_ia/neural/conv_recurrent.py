"""ConvRecurrentQNetwork — Conv2D + LSTM + FC pour V2-ZY.

Architecture (pattern Hausknecht DRQN appliqué à CNN) :
    Input  : (seq, batch, in_channels * rows * cols)   ← obs flat from buffer
    Reshape: (seq*batch, in_channels, rows, cols)
    Conv2d block → ReLU
    Flatten → (seq*batch, conv_out_channels * rows * cols)
    Reshape: (seq, batch, conv_features)
    LSTM(batch_first=False)
    FC(lstm_hidden → n_actions)
    Output : (seq, batch, n_actions)

Convention V2-Y : batch_first=False.

Voir spec : docs/superpowers/specs/2026-05-23-mw-ia-cnn-lstm-double-dqn-design.md §2
"""
from __future__ import annotations

import torch
from torch import nn


class ConvRecurrentQNetwork(nn.Module):
    """Conv block + LSTM + FC pour DRQN spatial.

    Accepte des observations 1D-flat depuis le buffer (compat
    SequenceReplayBuffer V2-Y) et reshape internement en 3D pour le Conv block.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        rows: int,
        cols: int,
        n_actions: int,
        conv_channels: tuple[int, ...] = (32, 64),
        kernel_size: int = 3,
        padding: int = 1,
        lstm_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.rows = rows
        self.cols = cols
        self.n_actions = n_actions
        self.lstm_hidden = lstm_hidden

        conv_layers: list[nn.Module] = []
        prev = in_channels
        for ch in conv_channels:
            conv_layers.append(nn.Conv2d(prev, ch, kernel_size=kernel_size, padding=padding))
            conv_layers.append(nn.ReLU(inplace=True))
            prev = ch
        self.conv = nn.Sequential(*conv_layers)
        self._conv_out_features = prev * rows * cols

        self.lstm = nn.LSTM(
            input_size=self._conv_out_features,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=False,
        )

        self.fc_out = nn.Linear(lstm_hidden, n_actions)

    def forward(
        self,
        obs_seq: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """obs_seq shape (seq, batch, in_channels * rows * cols). Retourne (q, hidden)."""
        seq_len, batch_size, _ = obs_seq.shape
        x = obs_seq.reshape(seq_len * batch_size, self.in_channels, self.rows, self.cols)
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        x = x.reshape(seq_len, batch_size, self._conv_out_features)
        x, new_hidden = self.lstm(x, hidden)
        q = self.fc_out(x)
        return q, new_hidden
