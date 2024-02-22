import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from tqdm.auto import tqdm, trange

from fdd_defense.defenders.base import BaseDefender


class AE(nn.Module):
    def __init__(
            self,
            num_sensors: int,
            window_size: int,
            bottleneck_size: int
        ):
        super().__init__()
        self.num_sensors = num_sensors
        self.window_size = window_size
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_sensors * window_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        bottleneck = []
        for i in range(bottleneck_size):
            bottleneck.append(nn.Linear(64, 64))
            bottleneck.append(nn.ReLU())
        self.bottleneck = nn.Sequential(*bottleneck)
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, num_sensors * window_size)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x.view(-1, self.window_size, self.num_sensors)


class AutoEncoderDefender(BaseDefender):
    def __init__(self, model, autoencoder=None, bottleneck_size=3):
        super().__init__(model)
        num_sensors = self.model.dataset.df.shape[1]
        window_size = self.model.window_size
        if autoencoder is None:
            autoencoder = AE(num_sensors, window_size, bottleneck_size)
        self.autoencoder = autoencoder
        self.optimizer = Adam(self.autoencoder.parameters())
        self.loss = nn.MSELoss(reduction='mean')
        self.autoencoder.train()
        print('Autoencoder training...')
        for e in trange(self.model.num_epochs, desc='Epochs ...'):
            losses = []
            for ts, _, label in tqdm(self.model.dataloader, desc='Steps ...', leave=False):
                ts = torch.FloatTensor(ts).to(self.model.device)
                autoencoder_ts = self.autoencoder(ts)
                loss = self.loss(autoencoder_ts, ts)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
            print(f'Epoch {e+1}, Loss: {sum(losses) / len(losses):.4f}')

    def predict(self, batch: np.ndarray):
        ts = torch.FloatTensor(batch).to(self.model.device)
        with torch.no_grad():
            def_batch = self.autoencoder(ts)
        return self.model.predict(def_batch)
