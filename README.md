# FDD Defense: Adversarial Attacks and Defenses on Fault Diagnosis and Detection models

## Introduction

The development of the smart manufacturing trend includes the integration of Artificial Intelligence technologies into industrial processes. One example of such implementation is deep learning models that diagnose the current state of a technological process. Recent studies have demonstrated that small data perturbations, named adversarial attacks, can significantly affect the correct predictions of such models. This fact is critical in industrial systems, where AI-based decisions can be made to manage physical equipment. `fdd-defense` helps to evaluate the robustness of technological process diagnosis models to adversarial attacks, as well as consider defense methods. 

`fdd-defense` is a python library with adversarial attacks on Fault Detection and Diagnostic (FDD) models and defense methods against adversarial attacks. This repository contains the original implementation of methods from the paper [Adversarial Attacks and Defenses in Fault Detection and Diagnosis: A Comprehensive Benchmark on the Tennessee Eastman Process](https://ieeexplore.ieee.org/abstract/document/10531068).

## Installing

To install `fdd-defense`, run the following command:
```
pip install git+https://github.com/AIRI-Institute/fdd-defense.git
```

## Usage

```python
from fdd_defense.models import MLP
from fdd_defense.attackers import FGSMAttacker
from fdd_defense.defenders import AdversarialTrainingDefender
from fdd_defense.utils import evaluate
from fddbenchmark import FDDDataset
from sklearn.preprocessing import StandardScaler

# Download and scale the TEP dataset
dataset = FDDDataset(name='reinartz_tep')
scaler = StandardScaler()
scaler.fit(dataset.df[dataset.train_mask])
dataset.df[:] = scaler.transform(dataset.df)

# Define and train a FDD model
model = MLP(
    window_size=50, 
    step_size=1, 
    device='cuda', 
    batch_size=128, 
    num_epochs=10
)
model.fit(dataset)

# Test the FDD model on original data without defense
defender = NoDefenceDefender(model)
attacker = NoAttacker(model, eps=epsilon)
accuracy = evaluate(defender, attacker)
print(f'Accuracy: {accuracy:.4f}')

# Test the FDD model under FGSM attack without defense
defender = NoDefenceDefender(model)
attacker = FGSMAttacker(defender, eps=epsilon)
accuracy = evaluate(defender, attacker)
print(f'Accuracy: {accuracy:.4f}')

# Test the FDD model under FGSM attack with Adversarial Training defense
defender = AdversarialTrainingDefender(model)
attacker = FGSMAttacker(defender, eps=epsilon)
accuracy = evaluate(defender, attacker)
print(f'Accuracy: {accuracy:.4f}')

```

## Implemented methods

### FDD models

| FDD model       | Reference |
|-----------------|-----------|
| Linear          |Pandya, D., Upadhyay, S. H., & Harsha, S. P. (2014). Fault diagnosis of rolling element bearing by using multinomial logistic regression and wavelet packet transform. Soft Computing, 18, 255-266.|
|Boosting         |Ruder, Sebastian. "An overview of gradient descent optimization algorithms." arXiv preprint arXiv:1609.04747 (2016).|
| MLP             |Khoualdia, T., Lakehal, A., Chelli, Z., Khoualdia, K., & Nessaib, K. (2021). Optimized multi layer perceptron artificial neural network based fault diagnosis of induction motor using vibration signals. Diagnostyka, 22.|
| GRU, TCN        |Lomov, Ildar, et al. "Fault detection in Tennessee Eastman process with temporal deep learning models." Journal of Industrial Information Integration 23 (2021): 100216.|

### Adversarial attacks

| Adversarial attack     | Reference |
|------------------------|-----------|
| Noise                  |Zhuo, Yue, Zhenqin Yin, and Zhiqiang Ge. "Attack and defense: Adversarial security of data-driven FDC systems." IEEE Transactions on Industrial Informatics 19.1 (2022): 5-19.|
| FGSM                   |Goodfellow, Ian J., Jonathon Shlens, and Christian Szegedy. "Explaining and harnessing adversarial examples." arXiv preprint arXiv:1412.6572 (2014).|
| PGD                    |Madry, Aleksander, et al. "Towards deep learning models resistant to adversarial attacks." arXiv preprint arXiv:1706.06083 (2017).|
| DeepFool               |Moosavi-Dezfooli, Seyed-Mohsen, Alhussein Fawzi, and Pascal Frossard. "Deepfool: a simple and accurate method to fool deep neural networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.|
| Carlini & Wagner       |Carlini, Nicholas, and David Wagner. "Towards evaluating the robustness of neural networks." 2017 ieee symposium on security and privacy (sp). Ieee, 2017.|
| Distillation black-box |Cui, Weiyu, et al. "Substitute model generation for black-box adversarial attack based on knowledge distillation." 2020 IEEE International Conference on Image Processing (ICIP). IEEE, 2020.|

### Defense methods

| Defense method          | Reference |
|-------------------------|-----------|
| Adversarial training    |Goodfellow, Ian J., Jonathon Shlens, and Christian Szegedy. "Explaining and harnessing adversarial examples." arXiv preprint arXiv:1412.6572 (2014).|
| Data quantization       |Guo, Chuan, et al. "Countering adversarial images using input transformations." arXiv preprint arXiv:1711.00117 (2017).|
| Gradient regularization |Finlay, Chris, and Adam M. Oberman. "Scaleable input gradient regularization for adversarial robustness." Machine Learning with Applications 3 (2021): 100017.|
| Defensive distillation  |Papernot, Nicolas, et al. "Distillation as a defense to adversarial perturbations against deep neural networks." 2016 IEEE symposium on security and privacy (SP). IEEE, 2016.| 
| ATQ  |Pozdnyakov, Vitaliy, et al. "Adversarial Attacks and Defenses in Fault Detection and Diagnosis: A Comprehensive Benchmark on the Tennessee Eastman Process." IEEE Open Journal of the Industrial Electronics Society (2024).| 

## Testing

To test the library, run the command `pytest tests` from the root directory.

## Citation

Please cite our paper as follows:

```
@article{pozdnyakov2024adversarial,
  title={Adversarial Attacks and Defenses in Fault Detection and Diagnosis: A Comprehensive Benchmark on the Tennessee Eastman Process},
  author={Pozdnyakov, Vitaliy and Kovalenko, Aleksandr and Makarov, Ilya and Drobyshevskiy, Mikhail and Lukyanov, Kirill},
  journal={IEEE Open Journal of the Industrial Electronics Society},
  year={2024},
  publisher={IEEE}
}
```