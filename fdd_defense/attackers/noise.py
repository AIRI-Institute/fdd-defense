from fdd_defense.attackers.base import BaseAttacker
import numpy as np


class NoiseAttacker(BaseAttacker):   
    def attack(self, ts, label):
        delta = self.eps * np.random.choice([1, -1], size=ts.shape)
        return ts + delta
