import numpy as np
from fdd_defense.defenders.base import BaseDefender
from fdd_defense.attackers import FGSMAttacker
from fdd_defense.utils import weight_reset
from tqdm.auto import tqdm, trange
import torch
from torch.optim import Adam


class AdversarialTrainingDefender(BaseDefender):
    def __init__(self, model, attacker=None, lambd=1):
        super().__init__(model)
        if attacker is None:
            attacker = FGSMAttacker(model, eps=0.1)
        self.attacker = attacker
        self.lambd = lambd
        self.model.model.apply(weight_reset)
        self.optimizer = Adam(self.model.model.parameters(), lr=self.model.lr)

        print('Adversarial training...')
        for e in trange(self.model.num_epochs, desc='Epochs ...'):
            losses = []
            for ts, _, label in tqdm(self.model.dataloader, desc='Steps ...', leave=False):
                batch_size = ts.shape[0]
                adv_ts = self.attacker.attack(ts, label)
                adv_ts = torch.FloatTensor(adv_ts).to(self.model.device)
                label = torch.LongTensor(label).to(self.model.device)
                ts = torch.FloatTensor(ts).to(self.model.device)
                _ts = torch.cat([ts, adv_ts])
                _logits = self.model.model(_ts)
                logits = _logits[:batch_size]
                adv_logits = _logits[batch_size:]
                real_loss = self.model.loss_fn(logits, label)
                adv_loss = self.model.loss_fn(adv_logits, label)
                loss = real_loss + self.lambd * adv_loss
                self.model.optimizer.zero_grad()
                loss.backward()
                self.model.optimizer.step()
                losses.append(loss.item())
                if self.model.is_test:
                    break
            print(f'Epoch {e+1}, Loss: {sum(losses) / len(losses):.4f}')
