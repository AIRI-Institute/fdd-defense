from fdd_defense.attackers.base import BaseAttacker
import torch
from tqdm.auto import trange

class DeepFoolAttacker(BaseAttacker):
    def __init__(
            self, 
            model: object, 
            eps: float,
            num_steps=10,
        ):
        super().__init__(model, eps)
        self.num_steps = num_steps
    
    def attack(self, ts, label):
        x = torch.FloatTensor(ts).to(self.model.device)
        y = torch.LongTensor(label).to(self.model.device)

        all_rows = range(len(x))
        x0 = x.clone()
        r_final = torch.zeros_like(x)
        num_classes = self.model.num_states
        self.model.model.train()
        for _ in range(self.num_steps):
            grads = []
            dists = []
            for k in range(num_classes):
                x.requires_grad = True
                
                logits = self.model.model(x)
                delta = logits[:, k] - logits[all_rows, y]
                delta.sum().backward()
                l1_norm = abs(x.grad).sum(dim=(1, 2))
                dist = abs(delta.detach()) / l1_norm
                grads.append(x.grad)
                dists.append(dist)
                x.grad = None
            grads = torch.stack(grads, dim=1)
            dists = torch.stack(dists, dim=1)
            dists[all_rows, y] = torch.inf
            min_dist, nearest_class = dists.min(dim=1)
            min_dist = torch.clamp(min_dist, max=self.eps)
            r = min_dist[:, None, None] * grads[all_rows, nearest_class].sign()
            correct_pred = logits.argmax(dim=1) == y
            in_limit = abs(r_final + r).amax(dim=(1, 2)) <= self.eps - 1e-6
            if (correct_pred & in_limit).sum() == 0:
                break
            r_final[correct_pred & in_limit] += r[correct_pred & in_limit]
            x = x0 + r_final
        return x.detach().cpu().numpy()
