from torch import nn
from fdd_defense.models.base import BaseTorchModel

class MLP(BaseTorchModel):
    def __init__(
            self, 
            window_size: int, 
            step_size: int, 
            batch_size=128,
            lr=0.001,
            num_epochs=10,
            is_test=False,
            device='cpu',
            hidden_dim=624,
        ):
        super().__init__(
            window_size, step_size, batch_size, lr, num_epochs, is_test, device,
        )
        self.hidden_dim = hidden_dim

    def _create_model(self, num_sensors, num_states):
        self.num_sensors = num_sensors
        self.num_states = num_states
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_sensors * self.window_size, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, num_states),
        )
