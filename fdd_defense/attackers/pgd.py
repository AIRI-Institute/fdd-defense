import numpy as np
from fdd_defense.attackers.base import BaseAttacker


class PGDAttacker(BaseAttacker):
    def __init__(
            self, 
            model: object, 
            eps: float,
            num_steps: int=10,
        ):
        super().__init__(model, eps)
        self.alpha = self.eps / num_steps
        self.num_steps = num_steps
    
    def attack(self, ts, label):
        super().attack(ts, label)
        delta = np.zeros_like(ts)
        for _ in range(self.num_steps):
            grad = self.model.get_grad(ts + delta, label)
            delta += self.alpha * np.sign(grad)
            delta = np.clip(delta, a_min=-self.eps, a_max=self.eps)
        return ts + delta
