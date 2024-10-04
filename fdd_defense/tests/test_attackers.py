from inspect import getmembers, isclass
from fdd_defense.models import MLP
from fdd_defense import attackers
from fddbenchmark import FDDDataset, FDDDataloader
from sklearn.preprocessing import minmax_scale
import numpy as np
import pytest
import torch

fdd_attackers = [f[1] for f in getmembers(attackers, isclass)]

class TestOnSmallTEP:
    def setup_class(self):
        self.dataset = FDDDataset(name='small_tep')
        self.dataset.df[:] = minmax_scale(self.dataset.df)
        self.eps = 0.01
        dataloader = FDDDataloader(
            dataframe=self.dataset.df,
            mask=self.dataset.train_mask,
            label=self.dataset.label, 
            window_size=10, 
            step_size=1, 
            use_minibatches=True, 
            batch_size=10,
        )
        for ts, _, label in dataloader:
            break
        self.ts = ts
        self.label = label
        
    @pytest.mark.parametrize("attacker", fdd_attackers)
    def test_base(self, attacker):
        torch.manual_seed(0)
        np.random.seed(0)
        fddmodel = MLP(window_size=10, step_size=1, is_test=True)
        fddmodel.fit(self.dataset)
        fdd_attacker = attacker(fddmodel, eps=self.eps)
        fdd_attacker.fit()
        adv_ts = fdd_attacker.attack(self.ts, self.label)
        eps = self.ts - adv_ts
        assert abs(eps).max() < self.eps + 1e-10
    
    @pytest.mark.parametrize("attacker", fdd_attackers)
    def test_loading(self, attacker):
        torch.manual_seed(0)
        np.random.seed(0)
        fddmodel = MLP(window_size=10, step_size=1, is_test=True)
        fddmodel.fit(self.dataset)
        fdd_attacker = attacker(fddmodel, eps=self.eps)
        fdd_attacker.fit()
        torch.save(fdd_attacker.model.model.state_dict(), 'weights.pt')
        fddmodel = MLP(window_size=10, step_size=1, is_test=True)
        num_sensors, num_states = self.dataset.df.shape[1], len(set(self.dataset.label))
        fddmodel.create_model(num_sensors, num_states)
        fdd_attacker = attacker(fddmodel, eps=self.eps)
        fdd_attacker.model.model.load_state_dict(
            torch.load('weights.pt', weights_only=True)
        )
        adv_ts = fdd_attacker.attack(self.ts, self.label)
        eps = self.ts - adv_ts
        assert abs(eps).max() < self.eps + 1e-10
