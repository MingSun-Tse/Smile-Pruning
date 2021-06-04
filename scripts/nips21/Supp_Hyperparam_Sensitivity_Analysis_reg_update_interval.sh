
# res56, pr=0.7, K_u=1
python main.py --method OPP -a resnet56 --lr_ft 0:0.01,60:0.001,90:0.0001 --epochs 120 --base_model_path Experiments/*SERVER138-20201209-110515/weights/checkpoint_best.pth --screen --dataset cifar10 --wd 0.0005 --batch_size 128 --batch_size_prune 128 --project OPPv5__resnet56__cifar10__pr0.7__lrft0.01__lwopp1000__hyperparam_sensitivity_analysis_update_reg_interval_1 --stage_pr [0,0.7,0.7,0.7,0] --update_reg_interval 1 --stabilize 10000 --opp_scheme 5 --lw_opp 1000

# res56, pr=0.7, K_u=5
python main.py --method OPP -a resnet56 --lr_ft 0:0.01,60:0.001,90:0.0001 --epochs 120 --base_model_path Experiments/*SERVER138-20201209-110515/weights/checkpoint_best.pth --screen --dataset cifar10 --wd 0.0005 --batch_size 128 --batch_size_prune 128 --project OPPv5__resnet56__cifar10__pr0.7__lrft0.01__lwopp1000__hyperparam_sensitivity_analysis_update_reg_interval_5 --stage_pr [0,0.7,0.7,0.7,0] --update_reg_interval 5 --stabilize 10000 --opp_scheme 5 --lw_opp 1000

# res56, pr=0.7, K_u=15
python main.py --method OPP -a resnet56 --lr_ft 0:0.01,60:0.001,90:0.0001 --epochs 120 --base_model_path Experiments/*SERVER138-20201209-110515/weights/checkpoint_best.pth --screen --dataset cifar10 --wd 0.0005 --batch_size 128 --batch_size_prune 128 --project OPPv5__resnet56__cifar10__pr0.7__lrft0.01__lwopp1000__hyperparam_sensitivity_analysis_update_reg_interval_15 --stage_pr [0,0.7,0.7,0.7,0] --update_reg_interval 15 --stabilize 10000 --opp_scheme 5 --lw_opp 1000

# res56, pr=0.7, K_u=20
python main.py --method OPP -a resnet56 --lr_ft 0:0.01,60:0.001,90:0.0001 --epochs 120 --base_model_path Experiments/*SERVER138-20201209-110515/weights/checkpoint_best.pth --screen --dataset cifar10 --wd 0.0005 --batch_size 128 --batch_size_prune 128 --project OPPv5__resnet56__cifar10__pr0.7__lrft0.01__lwopp1000__hyperparam_sensitivity_analysis_update_reg_interval_20 --stage_pr [0,0.7,0.7,0.7,0] --update_reg_interval 20 --stabilize 10000 --opp_scheme 5 --lw_opp 1000




# vgg19, pr=0.5, K_u=1
python main.py --method OPP -a vgg19 --lr_ft 0:0.01,60:0.001,90:0.0001 --epochs 120 --base_model_path Experiments/*SERVER138-20200530-145041/weights/checkpoint_best.pth --screen --dataset cifar100 --wd 0.0005 --batch_size 256 --batch_size_prune 256 --project OPPv5__vgg19__cifar100__pr0.5__lrft0.01__lwopp1000__hyperparam_sensitivity_analysis_update_reg_interval_1 --stage_pr 1-15:0.5 --update_reg_interval 1 --stabilize 10000 --opp_scheme 5 --lw_opp 1000

# vgg19, pr=0.5, K_u=5
python main.py --method OPP -a vgg19 --lr_ft 0:0.01,60:0.001,90:0.0001 --epochs 120 --base_model_path Experiments/*SERVER138-20200530-145041/weights/checkpoint_best.pth --screen --dataset cifar100 --wd 0.0005 --batch_size 256 --batch_size_prune 256 --project OPPv5__vgg19__cifar100__pr0.5__lrft0.01__lwopp1000__hyperparam_sensitivity_analysis_update_reg_interval_5 --stage_pr 1-15:0.5 --update_reg_interval 5 --stabilize 10000 --opp_scheme 5 --lw_opp 1000

# vgg19, pr=0.5, K_u=15
python main.py --method OPP -a vgg19 --lr_ft 0:0.01,60:0.001,90:0.0001 --epochs 120 --base_model_path Experiments/*SERVER138-20200530-145041/weights/checkpoint_best.pth --screen --dataset cifar100 --wd 0.0005 --batch_size 256 --batch_size_prune 256 --project OPPv5__vgg19__cifar100__pr0.5__lrft0.01__lwopp1000__hyperparam_sensitivity_analysis_update_reg_interval_15 --stage_pr 1-15:0.5 --update_reg_interval 15 --stabilize 10000 --opp_scheme 5 --lw_opp 1000

# vgg19, pr=0.5, K_u=20
python main.py --method OPP -a vgg19 --lr_ft 0:0.01,60:0.001,90:0.0001 --epochs 120 --base_model_path Experiments/*SERVER138-20200530-145041/weights/checkpoint_best.pth --screen --dataset cifar100 --wd 0.0005 --batch_size 256 --batch_size_prune 256 --project OPPv5__vgg19__cifar100__pr0.5__lrft0.01__lwopp1000__hyperparam_sensitivity_analysis_update_reg_interval_20 --stage_pr 1-15:0.5 --update_reg_interval 20 --stabilize 10000 --opp_scheme 5 --lw_opp 1000
