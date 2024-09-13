import numpy as np
from fdd_defense.models.base import BaseTorchModel
from fdd_defense.defenders.base import BaseDefender
from fdd_defense.utils import weight_reset
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from copy import deepcopy
from tqdm.auto import tqdm, trange
import torch


class CrossEntropyLoss(nn.Module):
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.nllloss = nn.NLLLoss()
    def forward(self, input, target):
        input = F.log_softmax(input / self.temp, dim=1)
        return -torch.sum(target * input, axis=1).mean()


class DistillationDefender(BaseDefender):
    def __init__(self, model, temp=100):
        super().__init__(model)
        assert issubclass(type(model), BaseTorchModel), "Distillation is applicable to neural networks only"
        
        self.model.model.apply(weight_reset)
        self.model.model.train()
        self.student = deepcopy(model)
        self.temp = temp

    def fit(self):
        loss_fn = CrossEntropyLoss(self.temp)
        num_states = len(set(self.model.dataset.label))
        print('Training a teacher...')
        optimizer = Adam(self.model.model.parameters(), lr=self.model.lr)
        for e in trange(self.model.num_epochs, desc='Epochs ...'):
            losses = []
            for ts, _, _label in tqdm(self.model.dataloader, desc='Steps ...', leave=False):
                ts = torch.FloatTensor(ts).to(self.model.device)
                label = F.one_hot(torch.LongTensor(_label), num_states).to(self.model.device)
                logits = self.model.model(ts)
                loss = loss_fn(logits, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                if self.model.is_test:
                    break
            print(f'Epoch {e+1}, Loss: {sum(losses) / len(losses):.4f}')

        
        print('Training a student...')
        optimizer = Adam(self.student.model.parameters(), lr=self.student.lr)
        for e in trange(self.student.num_epochs, desc='Epochs ...'):
            losses = []
            for ts, _, _ in tqdm(self.student.dataloader, desc='Steps ...', leave=False):
                ts = torch.FloatTensor(ts).to(self.student.device)
                with torch.no_grad():
                    label = self.model.model(ts)
                label = F.softmax(label, dim=1)
                logits = self.student.model(ts)
                loss = loss_fn(logits, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                if self.student.is_test:
                    break
            print(f'Epoch {e+1}, Loss: {sum(losses) / len(losses):.4f}')

    def predict(self, ts: np.ndarray):
        return self.student.predict(ts)
