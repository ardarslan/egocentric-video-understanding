import torch
import torch.nn as nn
from typing import Any, Dict


class NonLinearLayer(nn.Module):
    def __init__(
        self, cfg: Dict[str, Any], input_dimension: int, output_dimension: int
    ):
        super().__init__()
        self.cfg = cfg
        self.activation = nn.LeakyReLU(inplace=True)
        self.linear = nn.Linear(input_dimension, output_dimension, bias=False)
        self.normalization = nn.BatchNorm1d(num_features=output_dimension)
        self.dropout = nn.Dropout(cfg["dropout"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear(x)
        y = self.normalization(y)
        y = self.activation(y)
        if self.cfg["dropout"] > 0.0:
            y = self.dropout(y)
        return y


class OutputLayer(nn.Module):
    def __init__(
        self, cfg: Dict[str, Any], input_dimension: int, output_dimension: int
    ):
        super().__init__()
        self.cfg = cfg
        self.linear = nn.Linear(input_dimension, output_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear(x)
        return y
