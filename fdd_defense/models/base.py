import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from abc import ABC, abstractmethod
from fddbenchmark import FDDDataset, FDDDataloader
from tqdm.auto import tqdm, trange

class BaseModel(ABC):
    def __init__(self, window_size: int, step_size: int, is_test: bool, device: str):
        self.model = None
        self.window_size = window_size
        self.step_size = step_size
        self.is_test = is_test
        self.device = device

    @abstractmethod
    def fit(self, dataset: FDDDataset):
        self.dataset = dataset
    
    @abstractmethod
    def predict(self, ts: np.ndarray) -> np.ndarray:
        pass


class BaseTorchModel(BaseModel, ABC):
    def __init__(
            self, 
            window_size: int, 
            step_size: int, 
            batch_size: int,
            lr: float,
            num_epochs: int,
            is_test: bool,
            device: str,
        ):
        super().__init__(window_size, step_size, is_test, device)
        self.loss_fn = None
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.device = device

    def _train_nn(self):
        self.model.train()
        self.model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)

        self.dataloader = FDDDataloader(
            self.dataset.df,
            self.dataset.train_mask,
            self.dataset.label,
            window_size=self.window_size,
            step_size=self.step_size,
            use_minibatches=True,
            batch_size=self.batch_size,
            shuffle=True,
        )
        for e in trange(self.num_epochs, desc='Epochs ...'):
            losses = []
            for ts, _, label in tqdm(self.dataloader, desc='Steps ...', leave=False):
                label = torch.LongTensor(label).to(self.device)
                ts = torch.FloatTensor(ts).to(self.device)
                logits = self.model(ts)
                loss = self.loss_fn(logits, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                if self.is_test:
                    break
            print(f'Epoch {e+1}, Loss: {sum(losses) / len(losses):.4f}')

    def predict(self, ts: np.ndarray) -> np.ndarray:
        super().predict(ts)
        self.model.eval()
        self.model.to(self.device)
        ts = torch.FloatTensor(ts).to(self.device)
        with torch.no_grad():
            logits = self.model(ts)
        return logits.argmax(axis=1).cpu().numpy()
    
    def fit(self, dataset):
        super().fit(dataset=dataset)
        num_sensors, num_states = dataset.df.shape[1], len(set(dataset.label))
        self.num_sensors, self.num_states = num_sensors, num_states
        weight = torch.ones(num_states, device=self.device) * 0.5
        weight[1:] /= num_states
        self.loss_fn = nn.CrossEntropyLoss(weight=weight)
        self._create_model(num_sensors, num_states)
        self._train_nn()

    def create_model(self, num_sensors, num_states):
        self.num_sensors, self.num_states = num_sensors, num_states
        weight = torch.ones(num_states, device=self.device) * 0.5
        weight[1:] /= num_states
        self.loss_fn = nn.CrossEntropyLoss(weight=weight)
        self._create_model(num_sensors, num_states)

    def __call__(self, ts: torch.Tensor):
        return self.model(ts)

    def get_grad(self, ts: np.ndarray, label: np.ndarray) -> np.ndarray:
        self.model.train()
        self.model.to(self.device)
        self.model.zero_grad()
        ts = torch.FloatTensor(ts).to(self.device)
        label = torch.LongTensor(label).to(self.device)
        ts.requires_grad = True
        logits = self.model(ts)
        loss = self.loss_fn(logits, label)
        loss.backward()
        return ts.grad.data.cpu().numpy()
