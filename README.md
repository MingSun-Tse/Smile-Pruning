# Smile-Pruning
This repository is meant to provide a generic code base for neural network pruning, especially for **pruning at initialization** (PaI).

[[Survey](https://arxiv.org/abs/2103.06460) | [Paper Collection](https://github.com/MingSun-Tse/Awesome-Pruning-at-Initialization)]


## Step 1: Set up environment
- OS: Linux (Ubuntu 1404 and 1604 checked. It should be all right for most linux platforms. Windows and MacOS not checked.)
- python=3.6.9 (conda to manage environment is strongly suggested)
- All the dependant libraries are summarized in `requirements.txt`. Simply install them by `pip install -r requirements.txt`.
- CUDA and cuDNN

After the installlations, download the code:
```
git clone git@github.com:mingsun-tse/smile-pruning.git -b master
```

## Quick Start
The following script defines a whole IMP (iterative magnitude pruning) process in LTH (3 cycles). A quick try takes less than 1 min.
```python
CUDA_VISIBLE_DEVICES=0 python main.py --arch mlp_7_linear --activation linear --dataset mnist --batch_size 100 --project LTH__mlp_7_linear__mnist__wgweight__pr0.9__cycles3 --wg weight --hacksmile.ON --pipeline train:configs/LTH/train0.yaml,prune:configs/LTH/prune1.yaml,reinit:configs/LTH/reinit1.yaml,train:configs/LTH/train1.yaml,prune:configs/LTH/prune1.yaml,reinit:configs/LTH/reinit1.yaml,train:configs/LTH/train1.yaml,prune:configs/LTH/prune1.yaml,reinit:configs/LTH/reinit1.yaml,train:configs/LTH/train1.yaml --debug
```


## Acknowledgments
In this code we refer to the following implementations: [pytorch imagenet example](https://github.com/pytorch/examples/tree/master/imagenet), [rethinking-network-pruning](https://github.com/Eric-mingjie/rethinking-network-pruning), [EigenDamage-Pytorch](https://github.com/alecwangcq/EigenDamage-Pytorch), [pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10). Great thanks to them!

## Reference
Please cite this in your publication if our work helps your research:

    @article{wang2021recent,
      Author = {Wang, Huan and Qin, Can and Bai, Yue and Zhang, Yulun and Fu, Yun},
      Title = {Recent Advances on Neural Network Pruning at Initialization},
      Journal = {arXiv preprint arXiv:2103.06460},
      Year = {2021}
    }
    







