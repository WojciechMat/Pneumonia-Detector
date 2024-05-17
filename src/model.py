import torch
import torch.nn as nn


class PneumoniaModel(nn.Module):
    def __init__(self, config):
        super(PneumoniaModel, self).__init__()
        layers = []
        input_channels = 1

        # Convolutional layers
        for conv_layer in config.conv_layers:
            layers.append(
                nn.Conv2d(
                    input_channels,
                    conv_layer.out_channels,
                    kernel_size=conv_layer.kernel_size,
                    stride=1,
                    padding=1,
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            input_channels = conv_layer.out_channels

        self.conv = nn.Sequential(*layers)

        # Calculate the size of the flattened features
        sample_input_size = config.input_shape
        sample_input = torch.randn(1, 1, sample_input_size[0], sample_input_size[1])
        flattened_size = self._get_flattened_size(sample_input)

        # Fully connected layers
        dense_layers = []
        input_features = flattened_size
        for units in config.dense_layers:
            dense_layers.append(nn.Linear(input_features, units))
            dense_layers.append(nn.ReLU())
            dense_layers.append(nn.Dropout(0.1))
            input_features = units

        dense_layers.append(nn.Linear(input_features, 1))
        dense_layers.append(nn.Sigmoid())

        self.fc = nn.Sequential(*dense_layers)

    def _get_flattened_size(self, x):
        x = self.conv(x)
        return x.view(x.size(0), -1).size(1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
