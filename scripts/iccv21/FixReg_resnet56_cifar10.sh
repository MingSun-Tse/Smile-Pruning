
# ************************ LR 0.01 ************************
python main.py --method GReg-1 -a resnet56 --lr_ft 0:0.01,60:0.001,90:0.0001 --epochs 120 --base_model_path Experiments/teacher-resnet56-cifar10-wd0.0005-imagenetmean-bs128_SERVER138-20201209-110515/weights/checkpoint_best.pth --screen --dataset cifar10 --wd 0.0005 --batch_size 128 --batch_size_prune 128 --project FixReg__resnet56__cifar10__pr0.95_stabilize50000_lrft0.01 --stage_pr [0,0.95,0.95,0.95,0] --update_reg_interval 1 --reg_granularity_prune 1 --stabilize 50000 -j 2

python main.py --method GReg-1 -a resnet56 --lr_ft 0:0.01,60:0.001,90:0.0001 --epochs 120 --base_model_path Experiments/teacher-resnet56-cifar10-wd0.0005-imagenetmean-bs128_SERVER138-20201209-110515/weights/checkpoint_best.pth --screen --dataset cifar10 --wd 0.0005 --batch_size 128 --batch_size_prune 128 --project FixReg__resnet56__cifar10__pr0.5_stabilize50000_lrft0.01 --stage_pr [0,0.5,0.5,0.5,0] --update_reg_interval 1 --reg_granularity_prune 1 --stabilize 50000 -j 2

python main.py --method GReg-1 -a resnet56 --lr_ft 0:0.01,60:0.001,90:0.0001 --epochs 120 --base_model_path Experiments/teacher-resnet56-cifar10-wd0.0005-imagenetmean-bs128_SERVER138-20201209-110515/weights/checkpoint_best.pth --screen --dataset cifar10 --wd 0.0005 --batch_size 128 --batch_size_prune 128 --project FixReg__resnet56__cifar10__pr0.3_stabilize50000_lrft0.01 --stage_pr [0,0.3,0.3,0.3,0] --update_reg_interval 1 --reg_granularity_prune 1 --stabilize 50000 -j 2

python main.py --method GReg-1 -a resnet56 --lr_ft 0:0.01,60:0.001,90:0.0001 --epochs 120 --base_model_path Experiments/teacher-resnet56-cifar10-wd0.0005-imagenetmean-bs128_SERVER138-20201209-110515/weights/checkpoint_best.pth --screen --dataset cifar10 --wd 0.0005 --batch_size 128 --batch_size_prune 128 --project FixReg__resnet56__cifar10__pr0.7_stabilize50000_lrft0.01 --stage_pr [0,0.7,0.7,0.7,0] --update_reg_interval 1 --reg_granularity_prune 1 --stabilize 50000 -j 2

python main.py --method GReg-1 -a resnet56 --lr_ft 0:0.01,60:0.001,90:0.0001 --epochs 120 --base_model_path Experiments/teacher-resnet56-cifar10-wd0.0005-imagenetmean-bs128_SERVER138-20201209-110515/weights/checkpoint_best.pth --screen --dataset cifar10 --wd 0.0005 --batch_size 128 --batch_size_prune 128 --project FixReg__resnet56__cifar10__pr0.9_stabilize50000_lrft0.01 --stage_pr [0,0.9,0.9,0.9,0] --update_reg_interval 1 --reg_granularity_prune 1 --stabilize 50000 -j 2

# ************************ LR 0.001 ************************
# directly finetune the just pruned model using LR 0.001)

# [FixReg__resnet56__cifar10__pr0.3_stabilize50000_lrft0.01]
# [exp date: ['20211003', '20211003', '20211003']]
# [115-Pytorch-ImageNet-Prune] 060939, 095850, 095858
# 93.93, 93.97, 93.74 -- 93.88 (0.10)
# 93.98, 93.97, 93.93 -- 93.96 (0.02)
# acc_time: [119, 119, 119]
# test_acc_just_pruned: 93.0433 (0.0624)
python main.py --method GReg-1 -a resnet56 --lr_ft 0:0.001,80:0.0001 --epochs 120 --screen --dataset cifar10 --wd 0.0005 --batch_size 128 --stage_pr [0,0.3,0.3,0.3,0] --directly_ft_weights Exp*/*20211003-060939/weights/checkpoint_just_finished_prune.pth --project FixReg__resnet56__cifar10__pr0.3_stabilize50000_lrft0.001

python main.py --method GReg-1 -a resnet56 --lr_ft 0:0.001,80:0.0001 --epochs 120 --screen --dataset cifar10 --wd 0.0005 --batch_size 128 --stage_pr [0,0.3,0.3,0.3,0] --directly_ft_weights Exp*/*20211003-095850/weights/checkpoint_just_finished_prune.pth --project FixReg__resnet56__cifar10__pr0.3_stabilize50000_lrft0.001

python main.py --method GReg-1 -a resnet56 --lr_ft 0:0.001,80:0.0001 --epochs 120 --screen --dataset cifar10 --wd 0.0005 --batch_size 128 --stage_pr [0,0.3,0.3,0.3,0] --directly_ft_weights Exp*/*20211003-095858/weights/checkpoint_just_finished_prune.pth --project FixReg__resnet56__cifar10__pr0.3_stabilize50000_lrft0.001


# [FixReg__resnet56__cifar10__pr0.5_stabilize50000_lrft0.01]
# [exp date: ['20211003', '20211003', '20211003']]
# [115-Pytorch-ImageNet-Prune] 003436, 041839, 095927
# 93.59, 93.34, 93.43 -- 93.45 (0.10)
# 93.63, 93.48, 93.53 -- 93.55 (0.06)
# acc_time: [119, 119, 119]
# test_acc_just_pruned: 92.4300 (0.0356)
python main.py --method GReg-1 -a resnet56 --lr_ft 0:0.001,80:0.0001 --epochs 120 --screen --dataset cifar10 --wd 0.0005 --batch_size 128 --stage_pr [0,0.5,0.5,0.5,0] --directly_ft_weights Exp*/*20211003-003436/weights/checkpoint_just_finished_prune.pth --project FixReg__resnet56__cifar10__pr0.5_stabilize50000_lrft0.001

python main.py --method GReg-1 -a resnet56 --lr_ft 0:0.001,80:0.0001 --epochs 120 --screen --dataset cifar10 --wd 0.0005 --batch_size 128 --stage_pr [0,0.5,0.5,0.5,0] --directly_ft_weights Exp*/*20211003-041839/weights/checkpoint_just_finished_prune.pth --project FixReg__resnet56__cifar10__pr0.5_stabilize50000_lrft0.001

python main.py --method GReg-1 -a resnet56 --lr_ft 0:0.001,80:0.0001 --epochs 120 --screen --dataset cifar10 --wd 0.0005 --batch_size 128 --stage_pr [0,0.5,0.5,0.5,0] --directly_ft_weights Exp*/*20211003-095927/weights/checkpoint_just_finished_prune.pth --project FixReg__resnet56__cifar10__pr0.5_stabilize50000_lrft0.001

# [FixReg__resnet56__cifar10__pr0.7_stabilize50000_lrft0.01]
# [exp date: ['20211002', '20211003', '20211003']]
# [115-Pytorch-ImageNet-Prune] 115129, 022256, 061300
# 92.10, 92.32, 92.43 -- 92.28 (0.14)
# 92.25, 92.46, 92.43 -- 92.38 (0.09)
# acc_time: [119, 119, 119]
# test_acc_just_pruned: 90.8167 (0.0850)
python main.py --method GReg-1 -a resnet56 --lr_ft 0:0.001,80:0.0001 --epochs 120 --screen --dataset cifar10 --wd 0.0005 --batch_size 128 --stage_pr [0,0.7,0.7,0.7,0] --directly_ft_weights Exp*/*20211002-115129/weights/checkpoint_just_finished_prune.pth --project FixReg__resnet56__cifar10__pr0.7_stabilize50000_lrft0.001

python main.py --method GReg-1 -a resnet56 --lr_ft 0:0.001,80:0.0001 --epochs 120 --screen --dataset cifar10 --wd 0.0005 --batch_size 128 --stage_pr [0,0.7,0.7,0.7,0] --directly_ft_weights Exp*/*20211003-022256/weights/checkpoint_just_finished_prune.pth --project FixReg__resnet56__cifar10__pr0.7_stabilize50000_lrft0.001

python main.py --method GReg-1 -a resnet56 --lr_ft 0:0.001,80:0.0001 --epochs 120 --screen --dataset cifar10 --wd 0.0005 --batch_size 128 --stage_pr [0,0.7,0.7,0.7,0] --directly_ft_weights Exp*/*20211003-061300/weights/checkpoint_just_finished_prune.pth --project FixReg__resnet56__cifar10__pr0.7_stabilize50000_lrft0.001


# [FixReg__resnet56__cifar10__pr0.95_stabilize50000_lrft0.01]
# [exp date: ['20211002', '20211003', '20211003']]
# [115-Pytorch-ImageNet-Prune] 115210, 003415, 041618
# 85.81, 85.43, 85.95 -- 85.73 (0.22)
# 86.03, 85.63, 86.05 -- 85.90 (0.19)
# acc_time: [119, 119, 119]
# test_acc_just_pruned: 16.5200 (1.8671)
python main.py --method GReg-1 -a resnet56 --lr_ft 0:0.001,80:0.0001 --epochs 120 --screen --dataset cifar10 --wd 0.0005 --batch_size 128 --stage_pr [0,0.95,0.95,0.95,0] --directly_ft_weights Exp*/*20211002-115210/weights/checkpoint_just_finished_prune.pth --project FixReg__resnet56__cifar10__pr0.95_stabilize50000_lrft0.001

python main.py --method GReg-1 -a resnet56 --lr_ft 0:0.001,80:0.0001 --epochs 120 --screen --dataset cifar10 --wd 0.0005 --batch_size 128 --stage_pr [0,0.95,0.95,0.95,0] --directly_ft_weights Exp*/*20211003-003415/weights/checkpoint_just_finished_prune.pth --project FixReg__resnet56__cifar10__pr0.95_stabilize50000_lrft0.001

python main.py --method GReg-1 -a resnet56 --lr_ft 0:0.001,80:0.0001 --epochs 120 --screen --dataset cifar10 --wd 0.0005 --batch_size 128 --stage_pr [0,0.95,0.95,0.95,0] --directly_ft_weights Exp*/*20211003-041618/weights/checkpoint_just_finished_prune.pth --project FixReg__resnet56__cifar10__pr0.95_stabilize50000_lrft0.001

# [FixReg__resnet56__cifar10__pr0.9_stabilize50000_lrft0.01]
# [exp date: ['20211002', '20211003', '20211003']]
# [115-Pytorch-ImageNet-Prune] 115150, 022617, 080100
# 89.20, 89.13, 88.88 -- 89.07 (0.14)
# 89.39, 89.31, 89.01 -- 89.24 (0.16)
# acc_time: [119, 119, 119]
# test_acc_just_pruned: 62.0333 (3.5434)
python main.py --method GReg-1 -a resnet56 --lr_ft 0:0.001,80:0.0001 --epochs 120 --screen --dataset cifar10 --wd 0.0005 --batch_size 128 --stage_pr [0,0.9,0.9,0.9,0] --directly_ft_weights Exp*/*20211002-115150/weights/checkpoint_just_finished_prune.pth --project FixReg__resnet56__cifar10__pr0.9_stabilize50000_lrft0.001

python main.py --method GReg-1 -a resnet56 --lr_ft 0:0.001,80:0.0001 --epochs 120 --screen --dataset cifar10 --wd 0.0005 --batch_size 128 --stage_pr [0,0.9,0.9,0.9,0] --directly_ft_weights Exp*/*20211003-022617/weights/checkpoint_just_finished_prune.pth --project FixReg__resnet56__cifar10__pr0.9_stabilize50000_lrft0.001

python main.py --method GReg-1 -a resnet56 --lr_ft 0:0.001,80:0.0001 --epochs 120 --screen --dataset cifar10 --wd 0.0005 --batch_size 128 --stage_pr [0,0.9,0.9,0.9,0] --directly_ft_weights Exp*/*20211003-080100/weights/checkpoint_just_finished_prune.pth --project FixReg__resnet56__cifar10__pr0.9_stabilize50000_lrft0.001