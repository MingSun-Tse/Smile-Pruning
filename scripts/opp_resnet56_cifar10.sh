

BASIC="-a resnet56 --lr_ft 0:0.01,60:0.001,90:0.0001 --epochs 120 --base_model_path Experiments/teacher-resnet56-cifar10-wd0.0005-imagenetmean-bs128_SERVER138-20201209-110515/weights/checkpoint_best.pth --screen --dataset cifar10 --wd 0.0005 --batch_size 128 --batch_size_prune 128 --update_reg_interval 10 --stabilize 10000"

PR="0.95"
python main.py ${BASIC} --method OPP --project OPP__resnet56__cifar10__pr${PR}_lrft0.01__OrthReg_lw0.25_OppScheme4 --lw_opp 0.25 --stage_pr [0,${PR},${PR},${PR},0] --opp_scheme 4

PR="0.9"
python main.py ${BASIC} --method OPP --project OPP__resnet56__cifar10__pr${PR}_lrft0.01__OrthReg_lw0.25_OppScheme4 --lw_opp 0.25 --stage_pr [0,${PR},${PR},${PR},0] --opp_scheme 4

PR="0.7"
python main.py ${BASIC} --method OPP --project OPP__resnet56__cifar10__pr${PR}_lrft0.01__OrthReg_lw0.25_OppScheme4 --lw_opp 0.25 --stage_pr [0,${PR},${PR},${PR},0] --opp_scheme 4

PR="0.5"
python main.py ${BASIC} --method OPP --project OPP__resnet56__cifar10__pr${PR}_lrft0.01__OrthReg_lw0.25_OppScheme4 --lw_opp 0.25 --stage_pr [0,${PR},${PR},${PR},0] --opp_scheme 4

PR="0.3"
python main.py ${BASIC} --method OPP --project OPP__resnet56__cifar10__pr${PR}_lrft0.01__OrthReg_lw0.25_OppScheme4 --lw_opp 0.25 --stage_pr [0,${PR},${PR},${PR},0] --opp_scheme 4

