from fdd_defense.defenders.base import BaseDefender
from fdd_defense.utils import weight_reset
from tqdm.auto import tqdm, trange
import torch
from torch import nn


class RegularizationDefender(BaseDefender):
    """
    regularization = 'l2' or 'l1'
    the loss function is always redefined to cross entropy
    """
    def __init__(self, model, regularization='l2', lambd=1., h=0.01):
        super().__init__(model)
        self.lambd = lambd
        self.model.model.apply(weight_reset)
        self.model.model.train()
        self.regularization = regularization
        self.h = h
    
    def fit(self):
        print('Regularization training...')
        for e in trange(self.model.num_epochs, desc='Epochs ...'):
            losses = []
            for ts, _, label in tqdm(self.model.dataloader, desc='Steps ...', leave=False):
                label = torch.LongTensor(label).to(self.model.device)
                ts = torch.FloatTensor(ts).to(self.model.device)
                loss = self.regularized_loss(ts, label)
                self.model.optimizer.zero_grad()
                loss.backward()
                self.model.optimizer.step()
                losses.append(loss.item())
                if self.model.is_test:
                    break
            print(f'Epoch {e + 1}, Loss: {sum(losses) / len(losses):.4f}')

    def regularized_loss(self, ts, label):
        """
        algorinthm from paper "Scaleable input gradient regularization for adversarial robustness"
        link: https://arxiv.org/abs/1905.11468
        page 13
        """

        num_states = len(set(self.model.dataset.label))
        weight = torch.ones(num_states, device=self.model.device) * 0.5
        weight[1:] /= 20
        ce_loss = nn.CrossEntropyLoss(weight=weight, reduction='none')
        
        ts.requires_grad = True
        logits = self.model.model(ts)
        model_loss = ce_loss(logits, label).mean()
        gradient = torch.autograd.grad(model_loss, ts, create_graph=True)[0]
        ts.requires_grad = False
        if self.regularization == 'l1':
            gradient = torch.sign(gradient)
        grad_norm = gradient.norm(p=2, dim=(1, 2))

        grad_norm[grad_norm.abs() <= 1e-6] = float('inf')

        d = gradient / grad_norm[:, None, None]
        z = ts + self.h * d.detach()
        z_logits = self.model.model(z)

        ts_losses = ce_loss(logits, label)
        z_losses = ce_loss(z_logits, label)
        regularization = 1/(self.h**2) * torch.mean((z_losses - ts_losses) ** 2)

        return model_loss + self.lambd * regularization
