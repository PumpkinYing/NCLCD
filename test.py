import torch
import numpy as np

a = torch.zeros((5,5))
print(a)
a[1:2, :] = 1
print(a)
a[:, 1:2] = 1
print(a)