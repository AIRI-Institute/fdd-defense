import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler
from fddbenchmark import FDDDataset, FDDDataloader, FDDEvaluator

from fdd_defense.models import MLP, GRU, TCN
from fdd_defense.attackers import *
from fdd_defense.defenders import *


models = {'MLP': MLP, 'GRU': GRU, 'TCN': TCN}

attackers = {'NoiseAttacker': NoiseAttacker, 'FGSMAttacker': FGSMAttacker,
             'PGDAttacker': PGDAttacker, 'CarliniWagnerAttacker': CarliniWagnerAttacker,
             'DeepFoolAttacker': DeepFoolAttacker, 'DistillationBlackBoxAttacker': DistillationDefender,
             'NoAttackAttacker': NoAttacker}

defenders = {'AdversarialTrainingDefender': AdversarialTrainingDefender, 'DistillationDefender': DistillationDefender,
             'QuantizationDefender': QuantizationDefender, 'RegularizationDefender': RegularizationDefender,
             'DefensiveAutoencoderDefender': MLPAutoEncoderDefender, 'NoDefenceDefender': NoDefenceDefender}


def parse_args():
    parser = argparse.ArgumentParser(description='FDD Defense')
    parser.add_argument('--fddmodel', type=str, default='MLP')
    parser.add_argument('--attacker', type=str, default='NoAttackAttacker')
    parser.add_argument('--defender', type=str, default='NoDefenceDefender')
    return parser.parse_args()


def experiment():
    args = parse_args()

    #data preparation:
    window_size = 32
    dataset = FDDDataset(name='small_tep')
    scaler = StandardScaler()
    scaler.fit(dataset.df[dataset.train_mask])
    dataset.df[:] = scaler.transform(dataset.df)
    test_loader = FDDDataloader(
        dataframe=dataset.df,
        mask=dataset.test_mask,
        label=dataset.label,
        window_size=window_size,
        step_size=10,
        use_minibatches=True,
        batch_size=512,
    )
    evaluator = FDDEvaluator(step_size=1)

    #model creation:
    print('\n ' + args.fddmodel + ' model training:')
    fddmodel = models[args.fddmodel](window_size=window_size, step_size=1)
    fddmodel.fit(dataset)

    #attack
    print('\n attack: ' + args.attacker)

    #defense
    print('\n defense: ' + args.defender)
    defender = defenders[args.defender](fddmodel)

    #metrics
    attacker = attackers[args.attacker](model=defender, eps=0.05)
    preds = []
    labels = []
    for sample, index, label in test_loader:
        pred = fddmodel.predict(sample)
        adv_sample = attacker.attack(sample, pred)
        pred = defender.predict(adv_sample)
        preds.append(pd.Series(pred, index=index, name='pred'))
        labels.append(pd.Series(label, index=index, name='label'))
    preds = pd.concat(preds)
    labels = pd.concat(labels)
    metrics = evaluator.evaluate(labels, preds)
    fdr = metrics['confusion_matrix'].diagonal().sum() /  metrics['confusion_matrix'].sum()
    print('\n FDR:', fdr)


if __name__ == '__main__':
    experiment()