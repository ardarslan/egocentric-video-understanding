import torch.nn as nn


class Model(nn.Module):
    def __init__(
        self,
        num_activations: int,
        input_size: int,
        hidden_layer_size: str,
        output_size: int,
    ):
        super().__init__()
        self.num_activations = num_activations
        self.input_size = input_size
        self.output_size = output_size

        if hidden_layer_size == "min":
            self.hidden_layer_size = min(self.input_size, self.output_size)
        elif hidden_layer_size == "mean":
            self.hidden_layer_size = int((self.input_size + self.output_size) // 2)
        elif hidden_layer_size == "max":
            self.hidden_layer_size = max(self.input_size, self.output_size)

        self.linear_layers = []
        self.batch_norms = []

        for i in range(self.num_activations):
            if i == 0:
                self.linear_layers.append(
                    nn.Linear(self.input_size, self.hidden_layer_size)
                )
            else:
                self.linear_layers.append(
                    nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
                )
            self.batch_norms.append(nn.BatchNorm1d(self.hidden_layer_size))
        self.linear_layers.append(nn.Linear(self.hidden_layer_size, self.output_size))

    def forward(self, X):
        for i in range(self.num_activations):
            X = self.linear_layers[i](X)
            X = self.batch_norms[i](X)
            X = nn.LeakyReLU()(X)
        X = self.linear_layers[-1](X)
        return X
