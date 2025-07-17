r"""Neural networks"""

import torch.nn as nn
from typing import *

from torch import Tensor

class ConvEncoderDecoder_with_stride(nn.Module):
    def __init__(
        self, 
        in_features: int,
        hidden_channels: Sequence[int],
        kernel_sizes: Sequence[int],
        aux_features: int = 0,
        activation: Callable[[], nn.Module] = nn.ReLU,
    ):
        super().__init__()

        self.in_features = in_features
        self.forward_channels = [in_features + aux_features] + hidden_channels
        self.reverse_channels = hidden_channels[::-1] + [in_features]
        self.kernel_sizes = kernel_sizes

        encoder = []
        decoder = []

        for n_layer in range(len(hidden_channels)):
            # Encoder: downsample with stride=2
            encoder.append(
                nn.Conv2d(
                    in_channels=self.forward_channels[n_layer],
                    out_channels=self.forward_channels[n_layer + 1],
                    kernel_size=self.kernel_sizes[n_layer],
                    stride=2,
                    padding=(self.kernel_sizes[n_layer] - 1) // 2
                )
            )

            # Decoder: upsample with stride=2
            decoder.append(
                nn.ConvTranspose2d(
                    in_channels=self.reverse_channels[n_layer],
                    out_channels=self.reverse_channels[n_layer + 1],
                    kernel_size=self.kernel_sizes[n_layer],
                    stride=2,
                    padding=(self.kernel_sizes[n_layer] - 1) // 2,
                    output_padding=1  # makes sure spatial size doubles
                )
            )

            # Add BatchNorm + activation for all but last layer
            if n_layer < len(hidden_channels) - 1:
                encoder.append(nn.BatchNorm2d(self.forward_channels[n_layer + 1]))
                encoder.append(activation())
                decoder.append(nn.BatchNorm2d(self.reverse_channels[n_layer + 1]))
                decoder.append(activation())

        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon


