# alphaPruning
A sparse network training codebase for different learning settings related to sparse network with analysis.
## Why
Sparse network training involves many procedure such as training, pruning, finetuning, and reinitialization. Some works need iterative strategy like LTH using iterative magnitude pruning to find winning ticket.
In general, each of these procedures can be seen as one seperate module to achieve a complete pipeline. For example, conventional pruning == initialization+pretrain+prune+finetune, pruning at initialization == initialization+prune+finetune.
Inspired by this, we modulize each part to make a general codebase for convenient sparse network training research.
## How
The key config is ''pipeline''
## Analysis
Dynamic isometry, information plane, etc

