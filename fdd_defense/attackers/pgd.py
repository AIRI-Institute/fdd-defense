from fdd_defense.attackers.base import BaseAttacker
import torch

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
        delta = torch.zeros_like(ts)
        for _ in range(self.num_steps):
            grad = self.model.get_grad(ts + delta, label)
            delta += self.alpha * torch.sign(grad)
            delta = torch.clip(delta, min=-self.eps, max=self.eps)
        return ts + delta
