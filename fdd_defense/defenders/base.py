from abc import ABC, abstractmethod
import copy
import torch

class BaseDefender(ABC):  
    @abstractmethod
    def __init__(self, model: object):
        self.model = copy.deepcopy(model)
        pass

    def fit(self):
        pass
    
    def predict(self, ts):
        with torch.no_grad():
            return self.model.predict(ts)

    def get_grad(self, ts, label):
        return self.model.get_grad(ts, label)