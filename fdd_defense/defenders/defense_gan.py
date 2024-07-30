import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import RMSprop
from tqdm.auto import tqdm, trange
from fdd_defense.defenders.base import BaseDefender


# GRU MOMENT
class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]


class GRUGenerator(nn.Module):
    def __init__(
            self,
            num_sensors: int,
            window_size: int,
            noise_size: int = 256,
            ):
        self.num_sensors = num_sensors
        self.window_size = window_size
        self.noise_size = noise_size
        super().__init__()
        self.model = nn.Sequential(
            nn.GRU(self.noise_size, 128, num_layers=1, batch_first=True),
            SelectItem(0),
            nn.Linear(128, self.num_sensors),
        )

    def forward(self, x):
        return self.model(x)

    def generate(self, size):
        return self.__call__(torch.randn((size, self.window_size, self.noise_size), device=DEVICE))


class GRUDiscriminator(nn.Module):
    def __init__(
            self,
            num_sensors: int,
            window_size: int,
            ):
        super().__init__()
        self.num_sensors = num_sensors
        self.window_size = window_size
        self.model = nn.Sequential(
            nn.GRU(num_sensors, 128, num_layers=1, batch_first=True),
            SelectItem(0),
            nn.Linear(128, 128),
            nn.Flatten(),
            nn.Linear(128 * self.window_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


# MLP MOMENT
class MLPGenerator(nn.Module):
    def __init__(
            self,
            num_sensors: int,
            window_size: int,
            noise_size: int = 100):
        self.num_sensors = num_sensors
        self.window_size = window_size
        self.noise_size = noise_size
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(self.noise_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, self.num_sensors * self.window_size),
            nn.LayerNorm(self.num_sensors * self.window_size),
        )

    def forward(self, x):
        return self.model(x).view(-1, self.window_size, self.num_sensors)

    def generate(self, size, device):
        z = torch.randn((size, self.noise_size), device=device)
        return self.__call__(z)


class MLPDiscriminator(nn.Module):
    def __init__(
            self,
            num_sensors: int,
            window_size: int):
        super().__init__()
        self.num_sensors = num_sensors
        self.window_size = window_size
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(self.num_sensors * self.window_size),
            nn.Linear(self.num_sensors * self.window_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class DefenseGanDefender(BaseDefender):
    def __init__(self, model, random_restarts=10, optim_steps=200,
                 optim_lr=0.01, save_loss_history=False, mode="MLP"):
        super().__init__(model)
        # TODO ("MLP", "GRU") in mode else raise Exception
        self.random_restarts = random_restarts
        self.optim_steps = optim_steps
        self.optim_lr = optim_lr

        self.noise_len = 100

        self.device = self.model.device

        self.train_gan(save_loss_history)

        # from self.train_gan:
        # self.generator (eval mode), [self.gen_loss, self.discr_loss]

    def train_gan(self, save_loss_history=False):
        window_size = self.model.window_size  # expected 10
        num_sensors = self.model.dataset.df.shape[1]

        gen = MLPGenerator(num_sensors, window_size,
                        noise_size=self.noise_len).to(self.device)
        discr = MLPDiscriminator(num_sensors, window_size).to(self.device)

        gen_optimizer = RMSprop(gen.parameters(), lr=2e-6, maximize=True)
        discr_optimizer = RMSprop(discr.parameters(), lr=2e-6)

        discr_steps = 5

        discr_loss = []
        gen_loss = []
        num_epochs = 5
        for epoch in trange(num_epochs, desc='Epochs ...'):
            for true_data, _, label in tqdm(self.model.dataloader, desc='Steps ...', leave=False):
                # MINIMIZE DISCRIMINATOR
                true_data = torch.Tensor(true_data).to(self.device)
                batch_size = len(true_data)
                discr.train()
                gen.eval()
                for discr_step in range(discr_steps):
                    with torch.no_grad():
                        fake_data = gen.generate(batch_size, self.device)

                    pred = discr(fake_data).squeeze()
                    loss = F.binary_cross_entropy(pred, torch.zeros(batch_size, device=self.device))

                    pred = discr(true_data).squeeze()
                    loss += F.binary_cross_entropy(pred, torch.ones(batch_size, device=self.device))

                    discr_loss.append(loss.item())
                    discr_optimizer.zero_grad()
                    loss.backward()
                    discr_optimizer.step()

                gen.train()
                discr.eval()
                # MAXIMIZE GENERATOR
                fake_data = gen.generate(batch_size, self.device).to(self.device)
                pred = discr(fake_data).squeeze()
                loss = F.binary_cross_entropy(pred, torch.zeros(batch_size, device=self.device))
                gen_loss.append(loss.item())

                gen_optimizer.zero_grad()
                loss.backward()
                gen_optimizer.step()

        self.generator = gen
        if save_loss_history:
            self.gen_loss = gen_loss
            self.discr_loss = discr_loss
        self.generator.eval()
        self.generator.requires_grad_(False)

    def generate_similar(self, x: torch.Tensor) -> np.ndarray:
        # (H, W) -> (H, W)

        noise = torch.randn(size=(self.random_restarts, self.noise_len),
                            device=self.device, requires_grad=True)

        for optim_step in range(self.optim_steps):
            generated_data = self.generator(noise)
            dist = (generated_data - x).square().mean(dim=(1, 2))
            loss = dist.sum()
            noise.grad = None
            loss.backward()
            with torch.no_grad():
                noise -= self.optim_lr * noise.grad

        noise.requires_grad_(False)
        generated_data = self.generator(noise)
        dist = (generated_data - x).square().mean(dim=(1, 2))
        best_approx = generated_data[dist.argmin()]
        return best_approx.cpu().numpy()

    def predict(self, batch: np.ndarray) -> np.ndarray:
        # (N, H, W) -> (N,)
        batch = torch.FloatTensor(batch).to(self.device)
        approximations = []
        for x in tqdm(batch, desc='Iterating over batch ...', leave=False):
            approximations.append(self.generate_similar(x))
        return self.model.predict(np.stack(approximations))
