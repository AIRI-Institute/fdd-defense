from fdd_defense.attackers.base import BaseAttacker
import numpy as np
import torch

class NoiseAttacker(BaseAttacker):   
    def attack(self, ts, label):
        delta = self.eps * np.random.choice([1, -1], size=ts.shape)
        delta = torch.tensor(delta, dtype=torch.float32, device=self.model.device)
        return ts + delta
