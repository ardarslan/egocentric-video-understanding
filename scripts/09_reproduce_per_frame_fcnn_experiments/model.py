import torch
import torch.nn as nn
from layer import NonLinearLayer, OutputLayer


class MLP(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        hidden_dimension: int,
        num_nonlinear_layers: int,
    ):
        super().__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.hidden_dimension = hidden_dimension
        self.num_nonlinear_layers = num_nonlinear_layers
        self.layers = nn.ModuleList()
        if num_nonlinear_layers == 0:
            self.layers.append(
                OutputLayer(
                    input_dimension=self.input_dimension,
                    output_dimension=self.output_dimension,
                )
            )
        else:
            self.layers.append(
                NonLinearLayer(
                    input_dimension=self.input_dimension,
                    output_dimension=self.hidden_dimension,
                )
            )
            for _ in range(self.num_nonlinear_layers - 1):
                self.layers.append(
                    NonLinearLayer(
                        input_dimension=self.hidden_dimension,
                        output_dimension=self.hidden_dimension,
                    )
                )
            self.layers.append(
                OutputLayer(
                    input_dimension=self.hidden_dimension,
                    output_dimension=self.output_dimension,
                )
            )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            X = layer(X)
        return X
