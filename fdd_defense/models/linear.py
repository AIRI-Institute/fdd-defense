from torch import nn
from fdd_defense.models.base import BaseTorchModel


class LinearModel(BaseTorchModel):
    def __init__(
            self, 
            window_size, 
            step_size, 
            batch_size=64,
            lr=0.0001,
            num_epochs=1,
            is_test=False,
            device='cpu',
        ):
        super().__init__(window_size, step_size, batch_size, lr, num_epochs, is_test, device)
        
    def _create_model(self, num_sensors, num_states):
        self.num_sensors = num_sensors
        self.num_states = num_states
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_sensors * self.window_size, num_states),
        )
