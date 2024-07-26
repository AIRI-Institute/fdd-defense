import numpy as np
import random
from fdd_defense.defenders.base import BaseDefender
from fdd_defense.attackers import FGSMAttacker
from fdd_defense.utils import weight_reset
from tqdm.auto import trange, tqdm
import torch
from torch.optim import Adam


class ATQDefender(BaseDefender):
    def __init__(self, model, qbit=8):
        super().__init__(model)
        self.qbit = qbit
        self.min = model.dataset.df[model.dataset.train_mask].values.min(axis=0)
        self.max = model.dataset.df[model.dataset.train_mask].values.max(axis=0)
        self.min = self.min[None, None, :]
        self.max = self.max[None, None, :]
        self.eps = np.linspace(1e-6, 0.3, 20)
        
        print('ATQ training...')
        self.model.model.apply(weight_reset)
        self.model.model.train()
        for e in trange(self.model.num_epochs, desc='Epochs ...'):
            losses = []
            for ts, _, label in tqdm(self.model.dataloader, desc='Steps ...', leave=False):
                epsilon = random.choice(self.eps)
                attacker = FGSMAttacker(model, eps=epsilon)
                batch_size = ts.shape[0]
                adv_ts = attacker.attack(ts, label)
                label = torch.LongTensor(label).to(self.model.device)
                ts = torch.FloatTensor(self.quantize(ts)).to(self.model.device)
                adv_ts = torch.FloatTensor(self.quantize(adv_ts)).to(self.model.device)
                _ts = torch.cat([ts, adv_ts])
                _logits = self.model.model(_ts)
                logits = _logits[:batch_size]
                adv_logits = _logits[batch_size:]
                real_loss = self.model.loss_fn(logits, label)
                adv_loss = self.model.loss_fn(adv_logits, label)
                loss = real_loss + adv_loss
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
        def_batch = def_batch * scale +  self.min
        return def_batch

    def predict(self, batch: np.ndarray):
        def_batch = self.quantize(batch)
        return self.model.predict(def_batch)
