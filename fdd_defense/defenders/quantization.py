import numpy as np
from fdd_defense.defenders.base import BaseDefender
from fdd_defense.utils import weight_reset
from tqdm.auto import trange, tqdm
import torch
from torch.optim import Adam


class QuantizationDefender(BaseDefender):
    def __init__(self, model, qbit=8, min=None, max=None):
        super().__init__(model)
        self.qbit = qbit
        if min is None:
            self.min = self.model.dataset.df[self.model.dataset.train_mask].values.min(axis=0)
        if max is None:
            self.max = self.model.dataset.df[self.model.dataset.train_mask].values.max(axis=0)
        self.min = self.min[None, None, :]
        self.max = self.max[None, None, :]
        
    def fit(self):
        print('Quantization training...')
        self.model.model.apply(weight_reset)
        self.optimizer = Adam(self.model.model.parameters(), lr=self.model.lr)
        self.model.model.train()
        for e in trange(self.model.num_epochs, desc='Epochs ...'):
            losses = []
            for ts, _, label in tqdm(self.model.dataloader, desc='Steps ...', leave=False):
                label = torch.LongTensor(label).to(self.model.device)
                ts = torch.FloatTensor(self.quantize(ts)).to(self.model.device)
                logits = self.model.model(ts)
                loss = self.model.loss_fn(logits, label)
                self.model.optimizer.zero_grad()
                loss.backward()
                self.model.optimizer.step()
                losses.append(loss.item())
                if self.model.is_test:
                    break
            print(f'Epoch {e+1}, Loss: {sum(losses) / len(losses):.4f}')
        
    def quantize(self, batch: np.ndarray):
        scale = (self.max - self.min)
        scale[scale == 0] = 1
        batch_scaled = (batch - self.min) / scale
        def_batch = np.floor(batch_scaled * 2**self.qbit) / 2**self.qbit
        def_batch = def_batch * scale + self.min
        return def_batch

    def predict(self, batch: np.ndarray):
        def_batch = self.quantize(batch)
        return self.model.predict(def_batch)
