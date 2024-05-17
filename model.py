import torch
import torch.nn as nn


class PneumoniaModel(nn.Module):
    def __init__(self, config):
        super(PneumoniaModel, self).__init__()
        layers = []
        input_channels = 1

        # Create convolutional layers based on the configuration
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
            layers.append(nn.ReLU())  # Add ReLU activation
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))  # Add MaxPooling
            input_channels = conv_layer.out_channels  # Update input channels for next layer

        self.conv = nn.Sequential(*layers)  # Combine layers into a sequential module

        # Calculate the size of the flattened features after convolutional layers
        sample_input_size = config.input_shape  # Get input shape from config
        sample_input = torch.randn(1, 1, sample_input_size[0], sample_input_size[1])  # Create a sample input tensor
        flattened_size = self._get_flattened_size(sample_input)  # Compute the size after flattening

        # Create fully connected (dense) layers based on the configuration
        dense_layers = []
        input_features = flattened_size  # Start with the flattened feature size
        for units in config.dense_layers:
            dense_layers.append(nn.Linear(input_features, units))  # Add a linear layer
            dense_layers.append(nn.ReLU())  # Add ReLU activation
            dense_layers.append(nn.Dropout(0.1))  # Add dropout for regularization
            input_features = units  # Update input features for next layer

        dense_layers.append(nn.Linear(input_features, 1))  # Output layer
        dense_layers.append(nn.Sigmoid())  # Sigmoid activation for binary classification

        self.fc = nn.Sequential(*dense_layers)  # Combine dense layers into a sequential module

    def _get_flattened_size(self, x: torch.Tensor):
        # Pass the sample input through the convolutional layers to calculate the flattened size
        x = self.conv(x)
        return x.view(x.size(0), -1).size(1)  # Flatten and return the size

    def forward(self, x: torch.Tensor):
        # Define the forward pass
        x = self.conv(x)  # Pass through convolutional layers
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc(x)  # Pass through fully connected layers
        return x  # Return the final output
