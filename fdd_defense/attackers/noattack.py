from fdd_defense.attackers.base import BaseAttacker


class NoAttacker(BaseAttacker):  
    def attack(self, ts, label):
        super().attack(ts, label)
        return ts
