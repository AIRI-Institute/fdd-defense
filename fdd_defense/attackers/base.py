from abc import ABC, abstractmethod
import numpy as np
import copy

class BaseAttacker(ABC):
    """A base class of the attack.

    Attributes
    ----------
    model: object
        FDD model to be attacked
    eps: float
        data class with dataloaders

    Methods
    -------
    attack():
        Attack simulation
    """
    
    def __init__(self, model: object, eps: float):
        """Constructs all the necessary attributes for the attack object.

        Parameters
        ----------
            model : object
                FDD model to be attacked
            eps : float
                epsilon is the maximum absolute shift of the original data
        """
        assert eps > 0
        self.eps = eps
        self.model = copy.deepcopy(model)
    
    @abstractmethod
    def attack(self, ts: np.ndarray, label: np.ndarray) -> np.ndarray:
        """Attacks the data.

        Parameters
        ----------
            ts: np.ndarray
                attacked sensor data of the shape (batch size, 
                sequence length, number of sensors)
            label: np.ndarray
                label of sensor data of the shape (batch size, )
        """
        pass
    
    def fit(self):
        pass
