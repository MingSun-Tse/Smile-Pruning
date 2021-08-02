# lrft=0.01
python main.py -a alexnet --pretrained --method L1 --dataset imagenet --stage_pr [1-4:0.1] --lr_ft 0:0.01,30:0.001,60:0.0001,75:0.00001 --epochs 90 --project L1__alexnet__imagenet__pr0.1_lrft0.01 -j 6 --screen

python main.py -a alexnet --pretrained --method L1 --dataset imagenet --stage_pr [1-4:0.3] --lr_ft 0:0.01,30:0.001,60:0.0001,75:0.00001 --epochs 90 --project L1__alexnet__imagenet__pr0.3_lrft0.01 -j 6 --screen

python main.py -a alexnet --pretrained --method L1 --dataset imagenet --stage_pr [1-4:0.5] --lr_ft 0:0.01,30:0.001,60:0.0001,75:0.00001 --epochs 90 --project L1__alexnet__imagenet__pr0.5_lrft0.01 -j 6 --screen

python main.py -a alexnet --pretrained --method L1 --dataset imagenet --stage_pr [1-4:0.7] --lr_ft 0:0.01,30:0.001,60:0.0001,75:0.00001 --epochs 90 --project L1__alexnet__imagenet__pr0.7_lrft0.01 -j 6 --screen

python main.py -a alexnet --pretrained --method L1 --dataset imagenet --stage_pr [1-4:0.9] --lr_ft 0:0.01,30:0.001,60:0.0001,75:0.00001 --epochs 90 --project L1__alexnet__imagenet__pr0.9_lrft0.01 -j 6 --screen

python main.py -a alexnet --pretrained --method L1 --dataset imagenet --stage_pr [1-4:0.95] --lr_ft 0:0.01,30:0.001,60:0.0001,75:0.00001 --epochs 90 --project L1__alexnet__imagenet__pr0.95_lrft0.01 -j 6 --screen

# lrft=0.001
python main.py -a alexnet --pretrained --method L1 --dataset imagenet --stage_pr [1-4:0.1] --lr_ft 0:0.001,45:0.0001,68:0.00001 --epochs 90 --project L1__alexnet__imagenet__pr0.1_lrft0.001 -j 6 --screen

python main.py -a alexnet --pretrained --method L1 --dataset imagenet --stage_pr [1-4:0.3] --lr_ft 0:0.001,45:0.0001,68:0.00001 --epochs 90 --project L1__alexnet__imagenet__pr0.3_lrft0.001 -j 6 --screen

python main.py -a alexnet --pretrained --method L1 --dataset imagenet --stage_pr [1-4:0.5] --lr_ft 0:0.001,45:0.0001,68:0.00001 --epochs 90 --project L1__alexnet__imagenet__pr0.5_lrft0.001 -j 6 --screen

python main.py -a alexnet --pretrained --method L1 --dataset imagenet --stage_pr [1-4:0.7] --lr_ft 0:0.001,45:0.0001,68:0.00001 --epochs 90 --project L1__alexnet__imagenet__pr0.7_lrft0.001 -j 6 --screen

python main.py -a alexnet --pretrained --method L1 --dataset imagenet --stage_pr [1-4:0.9] --lr_ft 0:0.001,45:0.0001,68:0.00001 --epochs 90 --project L1__alexnet__imagenet__pr0.9_lrft0.001 -j 6 --screen

python main.py -a alexnet --pretrained --method L1 --dataset imagenet --stage_pr [1-4:0.95] --lr_ft 0:0.001,45:0.0001,68:0.00001 --epochs 90 --project L1__alexnet__imagenet__pr0.95_lrft0.001 -j 6 --screen

# scratch
python main.py -a alexnet --pretrained --method L1 --dataset imagenet --stage_pr [1-4:0.1] --lr_ft 0:0.1,30:0.01,60:0.001,90:0.0001,105:0.00001 --epochs 120 --project Scratch__alexnet__imagenet__pr0.1 -j 6 --screen

python main.py -a alexnet --pretrained --method L1 --dataset imagenet --stage_pr [1-4:0.3] --lr_ft 0:0.1,30:0.01,60:0.001,90:0.0001,105:0.00001 --epochs 120 --project Scratch__alexnet__imagenet__pr0.3 -j 6 --screen

python main.py -a alexnet --pretrained --method L1 --dataset imagenet --stage_pr [1-4:0.5] --lr_ft 0:0.1,30:0.01,60:0.001,90:0.0001,105:0.00001 --epochs 120 --project Scratch__alexnet__imagenet__pr0.5 -j 6 --screen

python main.py -a alexnet --pretrained --method L1 --dataset imagenet --stage_pr [1-4:0.7] --lr_ft 0:0.1,30:0.01,60:0.001,90:0.0001,105:0.00001 --epochs 120 --project Scratch__alexnet__imagenet__pr0.7 -j 6 --screen

python main.py -a alexnet --pretrained --method L1 --dataset imagenet --stage_pr [1-4:0.9] --lr_ft 0:0.1,30:0.01,60:0.001,90:0.0001,105:0.00001 --epochs 120 --project Scratch__alexnet__imagenet__pr0.9 -j 6 --screen

python main.py -a alexnet --pretrained --method L1 --dataset imagenet --stage_pr [1-4:0.95] --lr_ft 0:0.1,30:0.01,60:0.001,90:0.0001,105:0.00001 --epochs 120 --project Scratch__alexnet__imagenet__pr0.95 -j 6 --screen