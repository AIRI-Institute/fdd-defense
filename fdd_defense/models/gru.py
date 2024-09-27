import torch
from torch import nn
from fdd_defense.models.base import BaseTorchModel


class _GRUModule(nn.Module):
    def __init__(self, num_sensors, num_states, hidden_dim, num_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(num_sensors, hidden_dim, num_layers, batch_first=True)
        self.linear1 = nn.Linear(hidden_dim*num_layers, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, num_states)

    def forward(self, x):
        h = self.gru(x)[1].permute(1, 0, 2)
        h = h.reshape(h.size(0), -1)
        linear_out = self.linear1(h)
        linear_out = torch.relu(linear_out)
        linear_out = self.dropout(linear_out)
        out = self.linear2(linear_out)
        return out


class GRU(BaseTorchModel):
    def __init__(
            self, 
            window_size: int, 
            step_size: int, 
            batch_size=128,
            lr=0.001,
            num_epochs=5,
            is_test=False,
            device='cpu',
            hidden_dim=624,
            num_layers=2,
            dropout=0.,
        ):
        super().__init__(
            window_size, step_size, batch_size, lr, num_epochs, is_test, device,
        )
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

    def _create_model(self, num_sensors, num_states):
        self.num_sensors = num_sensors
        self.num_states = num_states
        self.model = _GRUModule(
            num_sensors, 
            num_states, 
            self.hidden_dim, 
            self.num_layers, 
            self.dropout
        )
