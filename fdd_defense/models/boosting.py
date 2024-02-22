import numpy as np
from catboost import CatBoostClassifier
from fddbenchmark import FDDDataset, FDDDataloader
from fdd_defense.models.base import BaseModel


class Boosting(BaseModel):
    def __init__(
            self, 
            window_size: int, 
            step_size: int, 
            iterations: int=1000,
            is_test: bool=False,
            device: str='CPU',
        ):
        super().__init__(window_size, step_size, is_test, device)
        self.iterations = iterations

    def fit(self, dataset: FDDDataset):
        super().fit(dataset)
        dataloader = FDDDataloader(
            dataset.df,
            dataset.train_mask,
            dataset.label,
            window_size=self.window_size,
            step_size=self.step_size,
        )
        for ts, _, label in dataloader:
            break
        if self.is_test:
            np.random.seed(0)
            idx = np.random.randint(ts.shape[0], size=10)
            ts, label = ts[idx], label[idx]
        num_states = len(set(label))
        weight = np.ones(num_states) * 0.5
        weight[1:] /= num_states
        self.model = CatBoostClassifier(
            iterations=1 if self.is_test else self.iterations,
            task_type=self.device,
            verbose=True,
            class_weights=weight,
        )
        ts = ts.reshape(ts.shape[0], -1)
        label = label.astype(int)
        self.model.fit(ts, label)

    def predict(self, ts: np.ndarray):
        super().predict(ts)
        ts = ts.reshape(ts.shape[0], -1)
        pred = self.model.predict(ts)[:, 0]
        return pred
