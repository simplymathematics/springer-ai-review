#!/bin/bash
mkdir models
python experiments.py --data cifar10 --train_size 50000 --max_iter 100 --verbose True
python experiments.py --data mnist --train_size 60000 --max_iter 100 --verbose True
while ls models/*.model;  do python  generate_attacks.py --data mnist  --attack_size 100 --max_iter 10 --verbose True  --threshold .03; sleep 600; done;
while ls models/*.model;  do python  generate_grads.py --data cifar  --attack_size 100 --max_iter 10 --verbose True  --threshold .03; sleep 600; done;
find . -name "*.model" | wc -l &&  echo "models to go. " &&  find . -name "*.attacked" | wc -l && echo "models completed"=