import numpy as np
import torch
from tqdm.auto import tqdm, trange

from fdd_defense.defenders.base import BaseDefender


# TODO: implement
class DefenseGanDefender(BaseDefender):
    def __init__(self, model, random_restarts=10, optim_steps=200):
        super().__init__(model)

    def generate_similar(self, x: torch.Tensor) -> np.ndarray:
        # (H, W) -> (H, W)
        return x.cpu().numpy()

    def predict(self, batch: np.ndarray) -> np.ndarray:
        # (N, H, W) -> (N,)
        batch = torch.FloatTensor(batch).to(self.device)
        approximations = []
        for x in tqdm(batch, desc='Iterating over batch ...', leave=False):
            approximations.append(self.generate_similar(x))
        return self.model.predict(np.stack(approximations))
