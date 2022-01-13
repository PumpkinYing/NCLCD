import os

# for instance_tau in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0, 2.0]:

for seed in range(20100, 20200):
    cmd = "python .\\train.py --instance_tau 0.7 --seed %d --lr 0.01 --data Cora"%(seed)
    os.system(cmd)

# for instance_tau in [0.2, 0.3, 0.4]:
#     for cluster_tau in [0.2, 0.3, 0.4]:
#         for lr in [0.001, 0.0001, 0.00001]:
#             for seed in [2333333]:
#                 for epoch in [200, 300, 400, 500]:
#                     # cmd = "python .\\train.py --instance_tau %f --cluster_tau %f" % (instance_tau, cluster_tau)
#                     cmd = "python .\\train.py --instance_tau %f --cluster_tau %f --seed %d --lr %f --epochs %d --data Cora" % (
#                         instance_tau, cluster_tau, seed, lr, epoch)
#                     os.system(cmd)

# for drop_1 in [0.1, 0.2, 0.3, 0.4]:
#     for drop_2 in [0.1, 0.2, 0.3, 0.4]:
#         for drop_feature_1 in [0.1, 0.2, 0.3, 0.4]:
#             for drop_feature_2 in [0.1, 0.2, 0.3, 0.4]:
#                 cmd = "python .\\train.py --instance_tau 0.4 --cluster_tau 0.4 --seed 2333333 --lr 0.0005 --epochs 300  \
#                     --data Cora --drop_edge_rate_1 %f --drop_edge_rate_2 %f --drop_feature_rate_1 %f --drop_feature_rate_2 %f" \
#                         %(drop_1, drop_2, drop_feature_1, drop_feature_2)
#                 os.system(cmd)