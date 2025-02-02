from fdd_defense.attackers.base import BaseAttacker
from fdd_defense.attackers import FGSMAttacker, NoiseAttacker, PGDAttacker, DeepFoolAttacker, CarliniWagnerAttacker
from torch.optim import Adam
from tqdm.auto import tqdm, trange
import copy


class DistillationBlackBoxAttacker(BaseAttacker):
    def __init__(
            self, 
            model: object, 
            eps: float,
            student: object=None,
        ):
        super().__init__(model, eps)
        if student is None:
            student = model
        self.student = copy.deepcopy(student)
        self.teacher = self.model
        self.attacker = FGSMAttacker(model=self.student, eps=self.eps)
    
    def attack(self, ts, label):
        adv_ts = self.attacker.attack(ts, label)
        return adv_ts

    def fit(self):
        self.student.fit(self.teacher.dataset)
        self.student.model.train()
        self.student.model.to(self.teacher.device)
        self.optimizer = Adam(self.student.model.parameters(), lr=self.student.lr)

        for e in trange(self.student.num_epochs, desc='Epochs ...'):
            losses = []
            for ts, _, label in tqdm(self.teacher.dataloader, desc='Steps ...', leave=False):
                label = self.teacher.predict(ts)
                logits = self.student.model(ts)
                loss = self.student.loss_fn(logits, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
            print(f'Epoch {e+1}, Loss: {sum(losses) / len(losses):.4f}')
