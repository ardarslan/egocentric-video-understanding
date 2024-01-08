import torch
import torch.nn as nn


class NonLinearLayer(nn.Module):
    def __init__(self, input_dimension: int, output_dimension: int):
        super().__init__()
        self.activation = nn.LeakyReLU(inplace=True)
        self.linear = nn.Linear(input_dimension, output_dimension, bias=False)
        self.normalization = nn.BatchNorm1d(num_features=output_dimension)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear(x)
        y = self.normalization(y)
        y = self.activation(y)
        y = self.dropout(y)
        return y


class OutputLayer(nn.Module):
    def __init__(self, input_dimension: int, output_dimension: int):
        super().__init__()
        self.linear = nn.Linear(input_dimension, output_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear(x)
        return y
