import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import RMSprop, Adam
from tqdm.auto import tqdm, trange

from fdd_defense.defenders.base import BaseDefender


class Generator(nn.Module):
    def __init__(
            self,
            num_sensors: int,
            window_size: int,
            noise_size: int = 256):
        self.num_sensors = num_sensors
        self.window_size = window_size
        self.noise_size = noise_size
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.LeakyReLU(True),
            nn.Linear(256, 256),
            nn.LeakyReLU(True),
            nn.Linear(256, 256),
            nn.LeakyReLU(True),
            nn.Linear(256, self.num_sensors * self.window_size),
        )

    def forward(self, x):
        return self.model(x).view(-1, self.window_size, self.num_sensors)


class Discriminator(nn.Module):
    def __init__(
            self,
            num_sensors: int,
            window_size: int):
        super().__init__()
        self.num_sensors = num_sensors
        self.window_size = window_size
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.num_sensors * self.window_size, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class DefenseGanDefender(BaseDefender):
    def __init__(self, model, random_restarts=10, optim_steps=1000,
                 optim_lr=0.01, save_loss_history=False):
        super().__init__(model)
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

        G = Generator(num_sensors=num_sensors, window_size=window_size).to(self.device)
        D = Discriminator(num_sensors=num_sensors, window_size=window_size).to(self.device)

        num_epochs = 300
        learning_rate = 1e-4

        G_losses = []
        D_losses = []
        iters = 0

        latent_size = 100  # (10, 10)
        batch_size = 512

        desc_steps = 1

        optim_G = torch.optim.AdamW(G.parameters(), lr=learning_rate)
        optim_D = torch.optim.RMSprop(D.parameters(), lr=learning_rate)

        criterion = nn.BCELoss()

        for epoch in trange(num_epochs, desc='Epochs ...'):
            for (data, _, _) in tqdm(self.model.dataloader, desc='Steps ...',
                                     leave=False):

                # 1. Обучим D: max log(D(x)) + log(1 - D(G(z)))
                D.train()

                for _ in range(desc_steps):
                    D.zero_grad()
                    data = torch.Tensor(data).to(self.device)
                    batch_size = len(data)
                    pred = D(data).view(-1)
                    true = torch.ones(batch_size).to(self.device)
                    loss_data = criterion(pred, true)
                    loss_data.backward()

                    z = torch.randn(batch_size, latent_size).to(self.device)
                    out = G(z)
                    pred = D(out.detach()).view(-1)
                    true = torch.zeros(batch_size).to(self.device)
                    loss_z = criterion(pred, true)
                    loss_z.backward()

                    D_loss = loss_z + loss_data

                    optim_D.step()

                # 2. Обучим G: max log(D(G(z)))
                G.train()
                G.zero_grad()

                D.eval()

                true = torch.ones(batch_size).to(self.device)
                pred = D(out).view(-1)
                loss = criterion(pred, true)
                loss.backward()

                G_loss = loss

                optim_G.step()

                D_losses.append(D_loss.item())
                G_losses.append(G_loss.item())

                iters += 1

        self.generator = G
        if save_loss_history:
            self.gen_loss = G_losses
            self.discr_loss = D_losses
        self.generator.eval()
        self.generator.requires_grad_(False)

    def generate_similar(self, x: torch.Tensor) -> np.ndarray:
        # (H, W) -> (H, W)

        noise = torch.randn(size=(self.random_restarts, self.noise_len),
                            device=self.device, requires_grad=True)
        optimizer = Adam([noise], lr=1e-3)

        for optim_step in range(self.optim_steps):
            generated_data = self.generator(noise)
            dist = (generated_data - x).square().mean(dim=(1, 2))
            loss = dist.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
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




class GRUDefenseGanDefender(BaseDefender):
    def __init__(self, model, random_restarts=1, optim_steps=1000,
                 optim_lr=1e-3, save_loss_history=False):
        super().__init__(model)
        self.random_restarts = random_restarts
        self.optim_steps = optim_steps
        self.optim_lr = optim_lr

        self.noise_len = 256

        self.device = self.model.device

        self.train_gan(save_loss_history)

        # from self.train_gan:
        # self.generator (eval mode), [self.gen_loss, self.discr_loss]

    def train_gan(self, save_loss_history=False):
        window_size = self.model.window_size  # expected 10
        num_sensors = self.model.dataset.df.shape[1]

        G = GRUGenerator(num_sensors=num_sensors, window_size=window_size).to(self.device)
        D = GRUDiscriminator(num_sensors=num_sensors, window_size=window_size).to(self.device)

        num_epochs = 1 # TODO
        learning_rate = 1e-4

        G_losses = []
        D_losses = []
        iters = 0

        latent_size = 256  # (10, 10)
        batch_size = 512

        desc_steps = 3

        optim_G = torch.optim.RMSprop(G.parameters(), lr=9 * learning_rate)
        optim_D = torch.optim.RMSprop(D.parameters(), lr=learning_rate)

        criterion = nn.BCELoss()

        for epoch in trange(num_epochs, desc='Epochs ...'):
            for (data, _, _) in tqdm(self.model.dataloader, desc='Steps ...',
                                     leave=False):

                # 1. Обучим D: max log(D(x)) + log(1 - D(G(z)))
                data = torch.Tensor(data).to(self.device)
                batch_size = len(data)

                for _ in range(desc_steps):

                    with torch.no_grad():
                        z = torch.randn((batch_size, window_size, self.noise_len), device=self.device)
                        fake_data = G(z)

                                
                    pred_fake = D(fake_data).squeeze()
                    loss_fake = F.binary_cross_entropy(pred_fake, torch.zeros(batch_size, device=self.device))

                    pred_true = D(data).squeeze()
                    loss_true = F.binary_cross_entropy(pred_true, torch.ones(batch_size, device=self.device))

                    loss = loss_fake + loss_true

                    optim_D.zero_grad()
                    loss.backward()

                    optim_G.zero_grad()
                    optim_D.step()
                    
                z = torch.randn((batch_size, window_size, self.noise_len), device=self.device)
                fake_data = G(z)
                pred = D(fake_data).squeeze()
                loss = F.binary_cross_entropy(pred, torch.zeros(batch_size, device=self.device))

                # 2. Обучим G: max log(D(G(z)))
                
                optim_G.zero_grad()
                loss.backward()
                optim_D.zero_grad()
                optim_G.step()

                iters += 1

        self.generator = G
        if save_loss_history:
            self.gen_loss = G_losses
            self.discr_loss = D_losses
        #self.generator.eval()
        self.generator.requires_grad_(False)

    def generate_similar(self, x: torch.Tensor) -> np.ndarray:
        # (H, W) -> (H, W)

        noise = torch.randn(size=(self.random_restarts, self.model.window_size, self.noise_len),
                            device=self.device, requires_grad=True)
        optimizer = Adam([noise], lr=1e-3)
        #print(noise.shape)

        for optim_step in range(self.optim_steps):
            generated_data = self.generator(noise)
            #print(generated_data.shape)
            dist = (generated_data - x).square().mean(dim=(1, 2))
            loss = dist.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        noise.requires_grad_(False)
        generated_data = self.generator(noise)
        dist = (generated_data - x).square().mean(dim=(1, 2))
        best_approx = generated_data[dist.argmin()]
        return best_approx.cpu().numpy()

    def predict(self, batch: np.ndarray) -> np.ndarray:
        # (N, H, W) -> (N,)
        batch = torch.FloatTensor(batch).to(self.device)
        approximations = []
        for x in tqdm(batch):
            approximations.append(self.generate_similar(x))
        return self.model.predict(np.stack(approximations))
