
# mlp_7_relu
python main.py --arch mlp_7_relu --dataset mnist --method OPP --opp_scheme v5 --base_model_path Experiments/Scratch__mlp_7_relu__mnist_SERVER138-20210310-160627/weights/checkpoint_best.pth --batch_size 100 --activation relu --project OPPv5__mlp_7_relu__mnist__wgfilter__pr0.9__lrft0.01__epoch90 --lr_ft 0:0.01,30:0.001,60:0.0001 --epochs 90 --stage_pr [0-5:0.9,6:0] --batch_size_prune 100 --update_reg_interval 10 --stabilize_reg_interval 10000


# lenet5_linear
python main.py --arch lenet5_linear --dataset mnist --method OPP --opp_scheme v5 --base_model_path Experiments/*-20210121-160140/weights/checkpoint_best.pth --batch_size 100 --activation linear --project OPPv5__lenet5_linear__mnist__wgfilter__pr0.9__lrft0.01__epoch90 --lr_ft 0:0.01,30:0.001,60:0.0001 --epochs 90 --stage_pr [0-2:0.9,3-4:0] --batch_size_prune 100 --update_reg_interval 10 --stabilize_reg_interval 10000

# lenet5
python main.py --arch lenet5 --dataset mnist --method OPP --opp_scheme v5 --base_model_path Experiments/*-20210310-164455/weights/checkpoint_best.pth --batch_size 100 --activation relu --project OPPv5__lenet5__mnist__wgfilter__pr0.9__lrft0.01__epoch90 --lr_ft 0:0.01,30:0.001,60:0.0001 --epochs 90 --stage_pr [0-2:0.9,3-4:0] --batch_size_prune 100 --update_reg_interval 10 --stabilize_reg_interval 10000