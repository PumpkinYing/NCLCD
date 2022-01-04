import os

for instance_tau in [0.2, 0.3, 0.4]:
    for cluster_tau in [0.2, 0.3, 0.4]:
        for lr in [0.001, 0.0001, 0.00001]:
            for seed in [2333333]:
                for epoch in [200, 300, 400, 500]:
                    # cmd = "python .\\train.py --instance_tau %f --cluster_tau %f" % (instance_tau, cluster_tau)
                    cmd = "python .\\train.py --instance_tau %f --cluster_tau %f --seed %d --lr %f --epochs %d --data Cora" % (
                        instance_tau, cluster_tau, seed, lr, epoch)
                    os.system(cmd)