

## cora
python train.py --data Cora --epochs 300 --cluster_tau 0.2 --instance_tau 0.7 --seed 2246 --model GCN --order 1 --theta 0.6

## citeseer
python train.py --data CiteSeer --epochs 300 --cluster_tau 0.4 --instance_tau 0.4 --seed 2736 --model GCN --order 1 --theta 0.9

## pubmed
python train.py --data PubMed --epochs 500 --cluster_tau 0.2 --instance_tau 0.5 --seed 238 --model GCN --order 1 --theta 0.8

## wiki
python train.py --data Wiki --epochs 500 --cluster_tau 0.9 --instance_tau 0.9 --seed 2131 --model GCN --order 1 --theta 0.6 