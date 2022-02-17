python main.py --method L1 -a vgg19 --dataset cifar100 --lr_ft 0:0.001,80:0.0001 --epochs 120 --base_model_path Experiments/*SERVER138-20200530-145041/weights/checkpoint_best.pth --wd 0.0005 --batch_size 128 --stage_pr [0,0.9,0.9,0.9,0] --project L1__vgg19__cifar100__pr0.9__lrft0.001

python main.py --method L1 -a vgg19 --dataset cifar100 --lr_ft 0:0.001,80:0.0001 --epochs 120 --base_model_path Experiments/*SERVER138-20200530-145041/weights/checkpoint_best.pth --wd 0.0005 --batch_size 128 --stage_pr [0,0.7,0.7,0.7,0] --project L1__vgg19__cifar100__pr0.7__lrft0.001

python main.py --method L1 -a vgg19 --dataset cifar100 --lr_ft 0:0.001,80:0.0001 --epochs 120 --base_model_path Experiments/*SERVER138-20200530-145041/weights/checkpoint_best.pth --wd 0.0005 --batch_size 128 --stage_pr [0,0.5,0.5,0.5,0] --project L1__vgg19__cifar100__pr0.5__lrft0.001

python main.py --method L1 -a vgg19 --dataset cifar100 --lr_ft 0:0.001,80:0.0001 --epochs 120 --base_model_path Experiments/*SERVER138-20200530-145041/weights/checkpoint_best.pth --wd 0.0005 --batch_size 128 --stage_pr [0,0.3,0.3,0.3,0] --project L1__vgg19__cifar100__pr0.3__lrft0.001

python main.py --method L1 -a vgg19 --dataset cifar100 --lr_ft 0:0.001,80:0.0001 --epochs 120 --base_model_path Experiments/*SERVER138-20200530-145041/weights/checkpoint_best.pth --wd 0.0005 --batch_size 128 --stage_pr [0,0.1,0.1,0.1,0] --project L1__vgg19__cifar100__pr0.1__lrft0.001