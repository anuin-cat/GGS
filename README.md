## GGS  (ICCV 2025)

Official implementation of “Enhancing Adversarial Transferability by Balancing Exploration and Exploitation with Gradient-Guided Sampling”. Supports single/ensemble, targeted/untargeted transfer attacks and evaluation across common CNN/ViT models.

The core implementation of this repository is inspired by, and will also be integrated into, https://github.com/Trustworthy-AI-Group/TransferAttack.

### Install

```bash
pip install -r requirements.txt
```

### Quick Start

Generate untargeted examples (single model):

```bash
# Targeted attack: add `--targeted`
# Ensemble attack: add `--ensemble`
python main.py --model resnet18 --batchsize 32
```



Evaluate ASR across models:

```bash
# Targeted attack: add `--targeted`
# Ensemble attack: add `--ensemble`
python main.py --eval  --model resnet18 --batchsize 16
```



### Run with adversarially trained models

- Download the corresponding converted weights (.npy) from [ylhz/tf_to_pytorch_model](https://github.com/ylhz/tf_to_pytorch_model) and place them under `models\\npy`.
- Open `main.py` and un-comment the line `for model_name, model in load_pretrained_model(...)` that includes `ens_model_paper` (the second line there) to evaluate adversarially trained/defended models.

### Citation

```bibtex
@inproceedings{Niu2025GGS,
  title     = {Enhancing Adversarial Transferability by Balancing Exploration and Exploitation with Gradient-Guided Sampling},
  author    = {Zenghao Niu and Weicheng Xie and Siyang Song and Zitong Yu and Feng Liu and Linlin Shen},
  booktitle = {ICCV},
  year      = {2025}
}
```