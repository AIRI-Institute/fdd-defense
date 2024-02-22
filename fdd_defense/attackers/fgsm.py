import numpy as np
from fdd_defense.attackers.base import BaseAttacker


class FGSMAttacker(BaseAttacker):  
    def attack(self, ts, label):
        super().attack(ts, label)
        grad = self.model.get_grad(ts, label)
        return ts + self.eps * np.sign(grad)
