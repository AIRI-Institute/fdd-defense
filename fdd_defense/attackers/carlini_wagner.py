"""
Carlini Wagner attack was proposed in Carlini, Nicholas, and David Wagner. 
"Towards evaluating the robustness of neural networks." 2017 ieee symposium 
on security and privacy (sp). Ieee, 2017.
The implementation is adopted from
https://github.com/Harry24k/adversarial-attacks-pytorch
https://github.com/bethgelab/foolbox
"""

from fdd_defense.attackers.base import BaseAttacker
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from tqdm.auto import trange

class CarliniWagnerAttacker(BaseAttacker):
    def __init__(
            self, 
            model: object, 
            eps: float, 
            num_steps: int=5,
            lr: float=0.01,
        ):
        super().__init__(model, eps)
        _min = self.model.dataset.df[self.model.dataset.train_mask].values.min()
        _max = self.model.dataset.df[self.model.dataset.train_mask].values.max()
        self.bounds = [_min, _max]
        self.num_steps = num_steps
        self.lr = lr

    def attack(self, _ts: np.ndarray, label: np.ndarray) -> np.ndarray:
        super().attack(_ts, label)
        ts = torch.FloatTensor(_ts).to(self.model.device)
        target = torch.LongTensor(label).to(self.model.device)
        adv_ts = ts.clone()
        w = inverse_tanh_space(adv_ts, self.bounds)
        w.requires_grad = True

        best_adv_ts = ts.clone()
        best_dist = 1e10 * torch.ones((len(ts))).to(self.model.device)

        optimizer = Adam([w], lr=self.lr)

        for _ in range(self.num_steps):
            adv_ts = tanh_space(w, self.bounds)
            dist = torch.abs(adv_ts - ts).amax(dim=(1, 2))
            
            self.model.model.train()
            logits = self.model.model(adv_ts)
            f_loss = f(logits, target, device=self.model.device)

            loss = dist.mean() + f_loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                pred = torch.argmax(logits, dim=1)
                neg_mask = (pred != target).float()
                
                mask = neg_mask*(best_dist > dist)
                best_dist = mask*dist + (1-mask)*best_dist

                mask = mask.view(-1, 1, 1)
                best_adv_ts = mask*adv_ts + (1-mask)*best_adv_ts

        best_adv_ts = torch.clamp(best_adv_ts, min=ts-self.eps, max=ts+self.eps)
        best_adv_ts = best_adv_ts.cpu().numpy()
        return best_adv_ts


def f(logits, target, device):
    """
    f objective function from the paper.
    """
    one_hot_label = torch.eye(logits.shape[1])[target].to(device)
    # find the max logit other than the target class
    other = torch.amax((1-one_hot_label)*logits, dim=1)
    # get the target class's logit
    real = torch.amax(one_hot_label*logits, dim=1)
    return torch.clamp((real-other), min=0)


def inverse_tanh_space(x, bounds):
    min_, max_ = bounds
    x = torch.clamp(x, min_, max_)
    a = (min_ + max_) / 2
    b = (max_ - min_) / 2
    x = (x - a) / b  # map from [min_, max_] to [-1, +1]
    x = x * 0.999999  # from [-1, +1] to approx. (-1, +1)
    x = x.arctanh()  # from (-1, +1) to (-inf, +inf)
    return x


def tanh_space(x, bounds):
    min_, max_ = bounds
    x = x.tanh()  # from (-inf, +inf) to (-1, +1)
    a = (min_ + max_) / 2
    b = (max_ - min_) / 2
    x = x * b + a  # map from (-1, +1) to (min_, max_)
    return x
