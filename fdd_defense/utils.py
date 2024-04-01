import numpy as np
from fddbenchmark import FDDDataloader
from tqdm.auto import tqdm

def weight_reset(model):
    """
    ref: https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/9
    """
    reset_parameters = getattr(model, "reset_parameters", None)
    if callable(reset_parameters):
        model.reset_parameters()

def accuracy(attacker, defender, step_size):
    test_loader = FDDDataloader(
        dataframe=attacker.model.dataset.df,
        mask=attacker.model.dataset.test_mask,
        label=attacker.model.dataset.label,
        window_size=attacker.model.window_size,
        step_size=step_size,
        use_minibatches=True,
        batch_size=512,
    )
    preds = []
    labels = []
    for sample, _, label in tqdm(test_loader):
        pred = attacker.model.predict(sample)
        adv_sample = attacker.attack(sample, pred)
        pred = defender.predict(adv_sample)
        preds.append(pred)
        labels.append(label)
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    return (preds == labels).sum() / len(preds)
