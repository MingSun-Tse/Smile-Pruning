# pr = 0.3
python main.py --method OPP -a resnet56 --lr_ft 0:0.001,80:0.0001 --epochs 120 --dataset cifar10 --wd 0.0005 --batch_size 128 --project OPPv5__resnet56__cifar10__pr0.3__lrft0.001__lwopp1000 --stage_pr [0,0.7,0.7,0.7,0] --directly_ft_weights Exp*/*-003523/weights/checkpoint_just_finished_prune.pth

python main.py --method OPP -a resnet56 --lr_ft 0:0.001,80:0.0001 --epochs 120 --dataset cifar10 --wd 0.0005 --batch_size 128 --project OPPv5__resnet56__cifar10__pr0.3__lrft0.001__lwopp1000 --stage_pr [0,0.7,0.7,0.7,0] --directly_ft_weights Exp*/*-003524/weights/checkpoint_just_finished_prune.pth

python main.py --method OPP -a resnet56 --lr_ft 0:0.001,80:0.0001 --epochs 120 --dataset cifar10 --wd 0.0005 --batch_size 128 --project OPPv5__resnet56__cifar10__pr0.3__lrft0.001__lwopp1000 --stage_pr [0,0.7,0.7,0.7,0] --directly_ft_weights Exp*/*-003527/weights/checkpoint_just_finished_prune.pth

# pr = 0.5
python main.py --method OPP -a resnet56 --lr_ft 0:0.001,80:0.0001 --epochs 120 --dataset cifar10 --wd 0.0005 --batch_size 128 --project OPPv5__resnet56__cifar10__pr0.5__lrft0.001__lwopp1000 --stage_pr [0,0.7,0.7,0.7,0] --directly_ft_weights Exp*/*-003258/weights/checkpoint_just_finished_prune.pth

python main.py --method OPP -a resnet56 --lr_ft 0:0.001,80:0.0001 --epochs 120 --dataset cifar10 --wd 0.0005 --batch_size 128 --project OPPv5__resnet56__cifar10__pr0.5__lrft0.001__lwopp1000 --stage_pr [0,0.7,0.7,0.7,0] --directly_ft_weights Exp*/*-003300/weights/checkpoint_just_finished_prune.pth

python main.py --method OPP -a resnet56 --lr_ft 0:0.001,80:0.0001 --epochs 120 --dataset cifar10 --wd 0.0005 --batch_size 128 --project OPPv5__resnet56__cifar10__pr0.5__lrft0.001__lwopp1000 --stage_pr [0,0.7,0.7,0.7,0] --directly_ft_weights Exp*/*-003305/weights/checkpoint_just_finished_prune.pth

# pr = 0.7
python main.py --method OPP -a resnet56 --lr_ft 0:0.001,80:0.0001 --epochs 120 --dataset cifar10 --wd 0.0005 --batch_size 128 --project OPPv5__resnet56__cifar10__pr0.7__lrft0.001__lwopp1000 --stage_pr [0,0.7,0.7,0.7,0] --directly_ft_weights Exp*/*-003233/weights/checkpoint_just_finished_prune.pth

python main.py --method OPP -a resnet56 --lr_ft 0:0.001,80:0.0001 --epochs 120 --dataset cifar10 --wd 0.0005 --batch_size 128 --project OPPv5__resnet56__cifar10__pr0.7__lrft0.001__lwopp1000 --stage_pr [0,0.7,0.7,0.7,0] --directly_ft_weights Exp*/*-003235/weights/checkpoint_just_finished_prune.pth

python main.py --method OPP -a resnet56 --lr_ft 0:0.001,80:0.0001 --epochs 120 --dataset cifar10 --wd 0.0005 --batch_size 128 --project OPPv5__resnet56__cifar10__pr0.7__lrft0.001__lwopp1000 --stage_pr [0,0.7,0.7,0.7,0] --directly_ft_weights Exp*/*-003237/weights/checkpoint_just_finished_prune.pth

# pr = 0.9
python main.py --method OPP -a resnet56 --lr_ft 0:0.001,80:0.0001 --epochs 120 --dataset cifar10 --wd 0.0005 --batch_size 128 --project OPPv5__resnet56__cifar10__pr0.9__lrft0.001__lwopp1000 --stage_pr [0,0.7,0.7,0.7,0] --directly_ft_weights Exp*/*-095205/weights/checkpoint_just_finished_prune.pth

python main.py --method OPP -a resnet56 --lr_ft 0:0.001,80:0.0001 --epochs 120 --dataset cifar10 --wd 0.0005 --batch_size 128 --project OPPv5__resnet56__cifar10__pr0.9__lrft0.001__lwopp1000 --stage_pr [0,0.7,0.7,0.7,0] --directly_ft_weights Exp*/*-180659/weights/checkpoint_just_finished_prune.pth

python main.py --method OPP -a resnet56 --lr_ft 0:0.001,80:0.0001 --epochs 120 --dataset cifar10 --wd 0.0005 --batch_size 128 --project OPPv5__resnet56__cifar10__pr0.9__lrft0.001__lwopp1000 --stage_pr [0,0.7,0.7,0.7,0] --directly_ft_weights Exp*/*-213040/weights/checkpoint_just_finished_prune.pth

# pr = 0.95
python main.py --method OPP -a resnet56 --lr_ft 0:0.001,80:0.0001 --epochs 120 --dataset cifar10 --wd 0.0005 --batch_size 128 --project OPPv5__resnet56__cifar10__pr0.95__lrft0.001__lwopp1000 --stage_pr 1-15:0.95 --directly_ft_weights Exp*/*-003204/weights/checkpoint_just_finished_prune.pth

python main.py --method OPP -a resnet56 --lr_ft 0:0.001,80:0.0001 --epochs 120 --dataset cifar10 --wd 0.0005 --batch_size 128 --project OPPv5__resnet56__cifar10__pr0.95__lrft0.001__lwopp1000 --stage_pr 1-15:0.95 --directly_ft_weights Exp*/*-003207/weights/checkpoint_just_finished_prune.pth

python main.py --method OPP -a resnet56 --lr_ft 0:0.001,80:0.0001 --epochs 120 --dataset cifar10 --wd 0.0005 --batch_size 128 --project OPPv5__resnet56__cifar10__pr0.95__lrft0.001__lwopp1000 --stage_pr 1-15:0.95 --directly_ft_weights Exp*/*-003213/weights/checkpoint_just_finished_prune.pth