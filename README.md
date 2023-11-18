# StarGAN-PyTorch

## Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Requirements](#requirements)
  - [From PyPI](#from-pypi)
  - [Local Install](#local-install)
- [All pretrained model weights](#all-pretrained-model-weights)
- [Test (e.g. CelebA-128x128)](#test-eg-celeba-128x128)
- [Train](#train)
- [Contributing](#contributing)
- [Credit](#credit)
  - [StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation](#stargan-unified-generative-adversarial-networks-for-multi-domain-image-to-image-translation)

## Introduction

This repository contains an op-for-op PyTorch reimplementation of [StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation](https://arxiv.org/abs/1711.09020v3).

## Getting Started

### Requirements

- Python 3.10+
- PyTorch 2.1.0+
- CUDA 11.8+
- Ubuntu 22.04+

### From PyPI

```bash
pip install stargan_pytorch -i https://pypi.org/simple
```

### Local Install

```bash
git clone https://github.com/Lornatang/StarGAN-PyTorch.git
cd StarGAN-PyTorch
pip install -r requirements.txt
pip install -e .
```

## All pretrained model weights

- [g_celeba128](https://huggingface.co/goodfellowliu/StarGAN-PyTorch/resolve/main/g_celeba128.pth.tar?download=true)
- [g_celeba256](https://huggingface.co/goodfellowliu/StarGAN-PyTorch/resolve/main/g_celeba256.pth.tar?download=true)
- [d_celeba128](https://huggingface.co/goodfellowliu/StarGAN-PyTorch/resolve/main/d_celeba128.pth.tar?download=true)
- [d_celeba256](https://huggingface.co/goodfellowliu/StarGAN-PyTorch/resolve/main/d_celeba256.pth.tar?download=true)

## Test (e.g. CelebA-128x128)

```shell
# Download g_celeba128 model weights to `./results/pretrained_models`
wget https://huggingface.co/goodfellowliu/StarGAN-PyTorch/resolve/main/g_celeba128.pth.tar?download=true -O ./results/pretrained_models/g_celeba128.pth.tar
python ./tools/test.py ./configs/CelebA128.yaml
# Result will be saved to `./results/test/celeba128`
```

<div align="center">
<img src="figure/celeba_128.jpg" width="768">
</div>

## Train

Please refer to `README.md` in the `data` directory for the method of making a dataset.

```shell
# If you want to train StarGAN-CelebA-128x128, run this command
python3 ./tools/train.py ./configs/CelebA128.yaml
# If you want to train StarGAN-CelebA-256x256, run this command
python3 ./tools/train.py ./configs/CelebA256.yaml
```

The training results will be saved to `./results/train/celeba128` or `./results/train/celeba256`.

## Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

## Credit

#### StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation

_Yunjey Choi, Minje Choi, Munyoung Kim, Jung-Woo Ha, Sunghun Kim, Jaegul Choo_ <br>

**Abstract** <br>
Recent studies have shown remarkable success in imageto-image translation for two domains. However, existing
approaches have limited scalability and robustness in handling more than two domains, since different models should
be built independently for every pair of image domains. To address this limitation, we propose StarGAN, a novel and
scalable approach that can perform image-to-image translations for multiple domains using only a single model.
Such a unified model architecture of StarGAN allows simultaneous training of multiple datasets with different domains
within a single network. This leads to StarGAN’s superior quality of translated images compared to existing models as
well as the novel capability of flexibly translating an input image to any desired target domain. We empirically demonstrate the effectiveness of our
approach on a facial attribute transfer and a facial expression synthesis tasks.

[[Paper]](https://arxiv.org/pdf/1711.09020v3) [[Code(PyTorch)]](https://github.com/yunjey/stargan)

```
@misc{choi2018stargan,
      title={StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation}, 
      author={Yunjey Choi and Minje Choi and Munyoung Kim and Jung-Woo Ha and Sunghun Kim and Jaegul Choo},
      year={2018},
      eprint={1711.09020},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```