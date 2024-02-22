from inspect import getmembers, isclass
from fdd_defense import models
from fdd_defense.models.base import BaseTorchModel
from fddbenchmark import FDDDataset, FDDDataloader
from sklearn.preprocessing import minmax_scale
import numpy as np
import pytest
import torch

fddmodels = [f[1] for f in getmembers(models, isclass)]

class TestOnSmallTEP:
    def setup_class(self):
        self.dataset = FDDDataset(name='small_tep')
        self.dataset.df[:] = minmax_scale(self.dataset.df)
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
        
    @pytest.mark.parametrize("model", fddmodels)
    def test_base(self, model):
        torch.manual_seed(0)
        np.random.seed(0)
        fddmodel = model(window_size=10, step_size=1, is_test=True)
        fddmodel.fit(self.dataset)
        pred = fddmodel.predict(self.ts)
        assert pred.shape == self.label.shape
        if fddmodel is BaseTorchModel:
            grad = fddmodel.get_grad(self.ts, self.label)
            assert grad.shape == self.ts.shape
