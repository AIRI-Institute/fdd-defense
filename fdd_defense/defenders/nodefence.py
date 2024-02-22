from fdd_defense.defenders.base import BaseDefender


class NoDefenceDefender(BaseDefender):
    def __init__(self, model):
        super().__init__(model)
