import os

for seed in range(2200,2300):
    for instance_tau in [0.9]:
        for cluster_tau in [0.9]:
            for epochs in [500]:
                for lr in [0.01]:
                    for theta in [0.6]:
                        for entropy in [0.1]:
                            cmd = "python train_dynamic.py --instance_tau %f --cluster_tau %f --seed %d --lr %f --data wiki \
                                --epochs %d --theta %f --entropy_weight %f --order 3" \
                                %(instance_tau, cluster_tau, seed, lr, epochs, theta, entropy)
                            os.system(cmd)

# for instance_tau in [0.2, 0.5, 0.7, 0.9]:
#     for cluster_tau in [0.2, 0.5, 0.7, 0.9]:
#         for lr in [0.01]:
#             for theta in [0.3, 0.5, 0.7, 0.9]:
#                 # for entropy in [0.1]:
#                 for epochs in [200, 300, 400, 500]:
#                     for data in ['wiki']:
#                         # cmd = "python .\\train.py --instance_tau %f --cluster_tau %f" % (instance_tau, cluster_tau)
#                             cmd = "python train.py --instance_tau %f --cluster_tau %f --lr %f --theta %f --entropy_weight 0.1 --data %s --epochs %d --order 1 --seed 23333 --model GCN" % (
#                                 instance_tau, cluster_tau, lr, theta, data, epochs)
#                             os.system(cmd)

# for instance_tau in [0.2, 0.5, 0.7, 0.9]:
#     for cluster_tau in [0.2, 0.5, 0.7, 0.9]:
#             cmd = "python train.py --instance_tau %f --cluster_tau %f --lr 0.01 --theta 0.7 --entropy_weight 0.1 --data Cora --order 1 --seed 2913 --epochs 300 --model GCN" % (instance_tau, cluster_tau)
#             os.system(cmd)

# for drop_1 in [0.1, 0.2, 0.3, 0.4]:
#     for drop_2 in [0.1, 0.2, 0.3, 0.4]:
#         for drop_feature_1 in [0.1, 0.2, 0.3, 0.4]:
#             for drop_feature_2 in [0.1, 0.2, 0.3, 0.4]:
#                 cmd = "python .\\train.py --instance_tau 0.4 --cluster_tau 0.4 --seed 2333333 --lr 0.0005 --epochs 300  \
#                     --data Cora --drop_edge_rate_1 %f --drop_edge_rate_2 %f --drop_feature_rate_1 %f --drop_feature_rate_2 %f" \
#                         %(drop_1, drop_2, drop_feature_1, drop_feature_2)
#                 os.system(cmd)