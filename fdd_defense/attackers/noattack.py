from fdd_defense.attackers.base import BaseAttacker


class NoAttacker(BaseAttacker):  
    def attack(self, ts, label):
        return ts
