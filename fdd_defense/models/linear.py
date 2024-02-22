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
        
    def fit(self, dataset):
        super().fit(dataset)
        num_sensors = self.dataset.df.shape[1]
        num_states = len(set(self.dataset.label))
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_sensors * self.window_size, num_states),
        )
        super()._train_nn()
