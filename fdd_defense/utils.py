import numpy as np
from tqdm.auto import tqdm

def weight_reset(model):
    """
    ref: https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/9
    """
    reset_parameters = getattr(model, "reset_parameters", None)
    if callable(reset_parameters):
        model.reset_parameters()

def accuracy(attacker, defender, loader):
    preds = []
    labels = []
    for sample, _, label in tqdm(loader):
        pred = attacker.model.predict(sample)
        adv_sample = attacker.attack(sample, pred)
        pred = defender.predict(adv_sample)
        preds.append(pred)
        labels.append(label)
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    return (preds == labels).sum() / len(preds)
