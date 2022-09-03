# Smile-Pruning
This repository is meant to provide a generic code base for neural network pruning, especially for **pruning at initialization** (PaI). (In preparation now, you may check our survey paper and paper collection below.)

[[Survey](https://arxiv.org/abs/2103.06460) | [Paper Collection](https://github.com/MingSun-Tse/Awesome-Pruning-at-Initialization)]

## Update Log
[09/03/2022] This repo is at a _slowly_ upgrading process right now due to the limited time of the main develpers (Huan and Yue, specifically). But in general, completing this code project is still on our schedule. Thanks for staying tuned!


## Set up environment
- OS: Linux (Ubuntu 1404 and 1604 checked. It should be all right for most linux platforms. Windows and MacOS not checked.)
- python=3.6.9 (conda to manage environment is strongly suggested)
- All the dependant libraries are summarized in `requirements_pt1.9.txt` (PyTorch 1.9 is used). Simply install them by `pip install -r requirements_pt1.9.txt`.
- CUDA and cuDNN

After the installlations, download the code:
```
git clone git@github.com:mingsun-tse/smile-pruning.git -b master
cd Smile-Pruning/src
```

## Quick Start
The following script defines a whole IMP (iterative magnitude pruning) process in LTH (3 cycles) with lenet5 on mnist. A quick try takes **less than 1 min**. Give it a shot!
```python
CUDA_VISIBLE_DEVICES=0 python main.py --arch lenet5 --dataset mnist --batch_size 100 --project LTH__lenet5__mnist__wgweight__pr0.9__cycles3 --pipeline train:configs/LTH/train0.yaml,prune:configs/LTH/prune1.yaml,reinit:configs/LTH/reinit1.yaml,train:configs/LTH/train1.yaml,prune:configs/LTH/prune1.yaml,reinit:configs/LTH/reinit1.yaml,train:configs/LTH/train1.yaml,prune:configs/LTH/prune1.yaml,reinit:configs/LTH/reinit1.yaml,train:configs/LTH/train1.yaml --debug
```

## Code Overview
We break up the (iterative) pruning process into 3 basic modules, corresponding to the 3 functions in `method_submodules`:
- `train.py` -- SGD training, which also is responsible for finetuning
- `prune.py` -- pruning
- `reinit.py` -- reinitialize a network. E.g., in LTH, after obtaning the masks (from the pretrained model), the weight values are rewound to the original initial values

Most pruning algorithm can be assembled by these 3 submodules, using the `--pipeline` argument -- which is the **ONLY** interface where a user defines a pruning process.


## Supported Pruning Methods, Datasets, Networks
We expect, given a kind of pruning pipeline (`--pipeline`), we can arbitrarily change the dataset (`--dataset`), network (`--arch`), within a choice pool. Currently, this code supports the following datasets and networks:
* datasets: mnist, fmnist, cifar10, cifar100, tinyimagenet, imagenet
* networks: lenet5, resnet56, resnet18/34/50

### How do I add my own dataset/network/pruning method?
(TODO)

## Acknowledgments
In this code we refer to the following implementations: [pytorch imagenet example](https://github.com/pytorch/examples/tree/master/imagenet), [rethinking-network-pruning](https://github.com/Eric-mingjie/rethinking-network-pruning), [EigenDamage-Pytorch](https://github.com/alecwangcq/EigenDamage-Pytorch), [pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10). Great thanks to them!

## Reference
If our paper/this paper collection/the code base helps your research/project, please generously consider to cite our paper. Thank you!

    @inproceedings{wang2022recent,
      Author = {Wang, Huan and Qin, Can and Bai, Yue and Zhang, Yulun and Fu, Yun},
      Title = {Recent Advances on Neural Network Pruning at Initialization},
      Booktitle = {IJCAI},
      Year = {2022}
    }
