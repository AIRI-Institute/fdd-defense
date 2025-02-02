from fdd_defense.attackers.base import BaseAttacker
import torch

class FGSMAttacker(BaseAttacker):  
    def attack(self, ts, label):
        grad = self.model.get_grad(ts, label)
        return ts + self.eps * torch.sign(grad)
