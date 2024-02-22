from fdd_defense.attackers.base import BaseAttacker
from fdd_defense.attackers import FGSMAttacker, NoiseAttacker, PGDAttacker, DeepFoolAttacker, CarliniWagnerAttacker
import torch
from torch import nn
from torch.optim import Adam
from tqdm.auto import tqdm, trange


class DistillationBlackBoxAttacker(BaseAttacker):
    def __init__(
            self, 
            model: object, 
            eps: float,
            student: object,
            base_attack: str
        ):
        super().__init__(model, eps)
        self.student = student
        self.student.fit(self.model.dataset)
        self.student.model.train()
        self.student.model.to(self.model.device)
        self.optimizer = Adam(self.student.model.parameters(), lr=self.student.lr)

        for e in trange(self.student.num_epochs, desc='Epochs ...'):
            losses = []
            for ts, _, label in tqdm(self.model.dataloader, desc='Steps ...', leave=False):
                ts = torch.FloatTensor(ts)
                label = self.model.predict(ts)
                label = torch.LongTensor(label).to(self.model.device)
                logits = self.student.model(ts.to(self.model.device))
                loss = self.student.loss_fn(logits, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
            print(f'Epoch {e+1}, Loss: {sum(losses) / len(losses):.4f}')
        self.attacker = FGSMAttacker(model=self.student, eps=eps)
        
    
    def attack(self, ts, label):
        super().attack(ts, label)
        adv_ts = self.attacker.attack(ts, label)
        return adv_ts