from abc import ABC, abstractmethod
import numpy as np
import copy

class BaseDefender(ABC):  
    @abstractmethod
    def __init__(self, model: object):
        self.model = copy.deepcopy(model)
        pass

    def fit(self):
        pass
    
    def predict(self, ts: np.ndarray):
        return self.model.predict(ts)

    def get_grad(self, ts: np.ndarray, label: np.ndarray):
        return self.model.get_grad(ts, label)