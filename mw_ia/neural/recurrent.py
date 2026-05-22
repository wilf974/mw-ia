"""RecurrentQNetwork (LSTM) pour V2-Y DRQN.

Voir spec : docs/superpowers/specs/2026-05-22-mw-ia-recurrent-network-design.md §2.1
"""
from __future__ import annotations

import torch
from torch import nn


class RecurrentQNetwork(nn.Module):
    """Réseau Q récurrent : Linear → ReLU → LSTM → Linear.

    Convention : batch_first=False, donc obs shape = (seq, batch, input_dim).
    Hidden = tuple (h, c) avec h.shape == c.shape == (1, batch, lstm_hidden)
    (1 layer LSTM). hidden=None → auto-init zéros (pattern PyTorch).
    """

    def __init__(
        self,
        input_dim: int,
        n_actions: int = 4,
        fc_hidden: int = 256,
        lstm_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.fc_hidden = fc_hidden
        self.lstm_hidden = lstm_hidden
        self.fc_in = nn.Linear(input_dim, fc_hidden)
        self.relu = nn.ReLU(inplace=True)
        self.lstm = nn.LSTM(fc_hidden, lstm_hidden, num_layers=1, batch_first=False)
        self.fc_out = nn.Linear(lstm_hidden, n_actions)

    def forward(
        self,
        obs_seq: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """obs_seq shape (seq, batch, input_dim). Retourne (q (seq, batch, n_actions), hidden).

        hidden=None → init zéros gérée par nn.LSTM.
        """
        x = self.fc_in(obs_seq)
        x = self.relu(x)
        x, hidden = self.lstm(x, hidden)
        q = self.fc_out(x)
        return q, hidden
