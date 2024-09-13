"""
The implementation of TCN is taken from the Darts library:
https://github.com/unit8co/darts
"""

from fdd_defense.models.base import BaseTorchModel
import torch.nn as nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(
        self,
        num_filters: int,
        kernel_size: int,
        dilation_base: int,
        dropout_fn,
        weight_norm: bool,
        nr_blocks_below: int,
        num_layers: int,
        input_size: int,
        target_size: int,
    ):
        """PyTorch module implementing a residual block module used in `TCNModule`.

        Parameters
        ----------
        num_filters
            The number of filters in a convolutional layer of the TCN.
        kernel_size
            The size of every kernel in a convolutional layer.
        dilation_base
            The base of the exponent that will determine the dilation on every level.
        dropout_fn
            The dropout function to be applied to every convolutional layer.
        weight_norm
            Boolean value indicating whether to use weight normalization.
        nr_blocks_below
            The number of residual blocks before the current one.
        num_layers
            The number of convolutional layers.
        input_size
            The dimensionality of the input time series of the whole network.
        target_size
            The dimensionality of the output time series of the whole network.

        Inputs
        ------
        x of shape `(batch_size, in_dimension, input_chunk_length)`
            Tensor containing the features of the input sequence.
            in_dimension is equal to `input_size` if this is the first residual block,
            in all other cases it is equal to `num_filters`.

        Outputs
        -------
        y of shape `(batch_size, out_dimension, input_chunk_length)`
            Tensor containing the output sequence of the residual block.
            out_dimension is equal to `output_size` if this is the last residual block,
            in all other cases it is equal to `num_filters`.
        """
        super().__init__()

        self.dilation_base = dilation_base
        self.kernel_size = kernel_size
        self.dropout_fn = dropout_fn
        self.num_layers = num_layers
        self.nr_blocks_below = nr_blocks_below

        input_dim = input_size if nr_blocks_below == 0 else num_filters
        output_dim = target_size if nr_blocks_below == num_layers - 1 else num_filters
        self.conv1 = nn.Conv1d(
            input_dim,
            num_filters,
            kernel_size,
            dilation=(dilation_base**nr_blocks_below),
        )
        self.conv2 = nn.Conv1d(
            num_filters,
            output_dim,
            kernel_size,
            dilation=(dilation_base**nr_blocks_below),
        )
        if weight_norm:
            self.conv1, self.conv2 = nn.utils.parametrizations.weight_norm(
                self.conv1
            ), nn.utils.parametrizations.weight_norm(self.conv2)

        if input_dim != output_dim:
            self.conv3 = nn.Conv1d(input_dim, output_dim, 1)

    def forward(self, x):
        residual = x

        # first step
        left_padding = (self.dilation_base**self.nr_blocks_below) * (
            self.kernel_size - 1
        )
        x = F.pad(x, (left_padding, 0))
        x = self.dropout_fn(F.relu(self.conv1(x)))

        # second step
        x = F.pad(x, (left_padding, 0))
        x = self.conv2(x)
        if self.nr_blocks_below < self.num_layers - 1:
            x = F.relu(x)
        x = self.dropout_fn(x)

        # add residual
        if self.conv1.in_channels != self.conv2.out_channels:
            residual = self.conv3(residual)
        x = x + residual

        return x


class TCNModule(nn.Module):
    def __init__(
        self,
        input_size: int,
        kernel_size: int,
        num_filters: int,
        num_layers: int,
        dilation_base: int,
        weight_norm: bool,
        target_size: int,
        dropout: float,
    ):

        """PyTorch module implementing a dilated TCN module used in `TCNModel`.


        Parameters
        ----------
        input_size
            The dimensionality of the input time series.
        target_size
            The dimensionality of the output time series.
        kernel_size
            The size of every kernel in a convolutional layer.
        num_filters
            The number of filters in a convolutional layer of the TCN.
        num_layers
            The number of convolutional layers.
        weight_norm
            Boolean value indicating whether to use weight normalization.
        dilation_base
            The base of the exponent that will determine the dilation on every level.
        dropout
            The dropout rate for every convolutional layer.

        Inputs
        ------
        x of shape `(batch_size, input_chunk_length, input_size)`
            Tensor containing the features of the input sequence.

        Outputs
        -------
        y of shape `(batch_size, input_chunk_length, target_size)`
            Tensor containing the predictions of the next 'output_chunk_length' points in the last
            'output_chunk_length' entries of the tensor. The entries before contain the data points
            leading up to the first prediction, all in chronological order.
        """

        super().__init__()

        # Defining parameters
        self.input_size = input_size
        self.n_filters = num_filters
        self.kernel_size = kernel_size
        self.target_size = target_size
        self.dilation_base = dilation_base
        self.dropout = nn.Dropout(p=dropout)

        # Building TCN module
        self.res_blocks_list = []
        for i in range(num_layers):
            res_block = ResidualBlock(
                num_filters,
                kernel_size,
                dilation_base,
                self.dropout,
                weight_norm,
                i,
                num_layers,
                self.input_size,
                target_size,
            )
            self.res_blocks_list.append(res_block)
        self.res_blocks = nn.ModuleList(self.res_blocks_list)

    def forward(self, x):
        # data is of size (batch_size, input_chunk_length, input_size)
        x = x.transpose(1, 2)
        for res_block in self.res_blocks_list:
            x = res_block(x)
        x = x.transpose(1, 2)[:, -1, :]
        return x


class TCN(BaseTorchModel):
    def __init__(
            self,
            window_size: int,
            step_size: int,
            batch_size=128,
            lr=0.001,
            num_epochs=10,
            is_test=False,
            device='cpu',
            kernel_size=3,
            num_filters=64,
            num_layers=2,
            dilation_base=2,
            weight_norm=True,
            dropout=0.,
        ):
        super().__init__(
            window_size, step_size, batch_size, lr, num_epochs, is_test, device,
        )

        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.num_layers = num_layers
        self.dilation_base = dilation_base
        self.weight_norm = weight_norm
        self.dropout = dropout


    def _create_model(self, num_sensors, num_states):
        self.model = TCNModule(
            input_size=num_sensors,
            kernel_size=self.kernel_size,
            num_filters=self.num_filters,
            num_layers=self.num_layers,
            dilation_base=self.dilation_base,
            weight_norm=self.weight_norm,
            target_size=num_states,
            dropout=self.dropout
        )
