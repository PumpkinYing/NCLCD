import os

for seed in range(233, 244):
    for instance_tau in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0, 2.0]:
        cmd = "python train.py --instance_tau %f --cluster_tau 0.2 --seed %d --lr 0.01 --data PubMed"%(instance_tau, seed)
        os.system(cmd)

# for instance_tau in [0.2, 0.5, 0.7, 0.9]:
#     for cluster_tau in [0.2, 0.5, 0.7]:
#         for lr in [0.01]:
#             for theta in [0.4, 0.7, 0.9]:
#                 for entropy in [0.1, 0.3, 0.5]:
#                     # cmd = "python .\\train.py --instance_tau %f --cluster_tau %f" % (instance_tau, cluster_tau)
#                     cmd = "python train.py --instance_tau %f --cluster_tau %f --lr %f --theta %f --entropy_weight %f --data PubMed" % (
#                         instance_tau, cluster_tau, lr, theta, entropy)
#                     os.system(cmd)

# for drop_1 in [0.1, 0.2, 0.3, 0.4]:
#     for drop_2 in [0.1, 0.2, 0.3, 0.4]:
#         for drop_feature_1 in [0.1, 0.2, 0.3, 0.4]:
#             for drop_feature_2 in [0.1, 0.2, 0.3, 0.4]:
#                 cmd = "python .\\train.py --instance_tau 0.4 --cluster_tau 0.4 --seed 2333333 --lr 0.0005 --epochs 300  \
#                     --data Cora --drop_edge_rate_1 %f --drop_edge_rate_2 %f --drop_feature_rate_1 %f --drop_feature_rate_2 %f" \
#                         %(drop_1, drop_2, drop_feature_1, drop_feature_2)
#                 os.system(cmd)