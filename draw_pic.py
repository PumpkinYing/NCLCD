import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import seaborn as sns
import palettable#python颜色库
from sklearn import datasets 


plt.rcParams['font.sans-serif']=['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

plt.figure(dpi=200, figsize=(10,6))

data = [[0.48375184638109303, 0.3529240944225857, 0.40195628947078177, 0.2977790445983128],
[0.42134416543574593, 0.34924757147090624, 0.3797652371330202, 0.25902383858064115],
[0.41986706056129985, 0.34701936805353206, 0.3399164255104091, 0.270918451374455],
[0.45679468242245197, 0.3552073301823322, 0.34881514268453645, 0.2968416177316965],
[0.6026587887740029, 0.4683487192987212, 0.5464234799605959, 0.4045903152878489],
[0.603397341211226, 0.46818216609773383, 0.520452782632779, 0.40481501540641895],
[0.6052437223042836, 0.45873563970016434, 0.5332219964870154, 0.4197321171148784],
[0.6418020679468243, 0.4671320363332857, 0.5795785590030506, 0.436194073441493],
[0.6333087149187593, 0.49772926384106553, 0.5796359501787466, 0.4259226559721891],
[0.6872230428360414, 0.5207315184016107, 0.6198174463920718, 0.4834879760410128],
[0.6683899556868538, 0.5189246537573229, 0.62490621647647, 0.48562767437891463],
[0.6838995568685377, 0.5200457886409778, 0.6231391288856718, 0.4866311387120013],
[0.6994091580502215, 0.5247617325501216, 0.6151994866758466, 0.49290921908075425],
[0.7141802067946824, 0.5313464084206948, 0.6293713809824036, 0.5027194035444453],
[0.7163958641063516, 0.5283682890167547, 0.6372234870540023, 0.5080896728355974],
[0.7256277695716395, 0.5321353481446326, 0.6957145691625041, 0.5140218804285653]]

# data = [[0.6222304283604135, 0.517653176132656, 0.6032092713098844, 0.4183000186744872],
# [0.6322008862629247, 0.5287925170120691, 0.6102080915248365, 0.4208233739902358],
# [0.7042097488921714, 0.5496109624264216, 0.640026988222288, 0.5227657659682614],
# [0.7208271787296898, 0.5522198102162703, 0.6444514667493132, 0.5167482913969271],
# [0.6388478581979321, 0.5302905804791569, 0.6113672210108072, 0.42062247943600567],
# [0.6366322008862629, 0.5297661233359794, 0.6088902387179556, 0.41319165635701327],
# [0.6362629246676514, 0.5262332781386309, 0.6076489303774494, 0.4102834040247133],
# [0.6366322008862629, 0.5269998778813386, 0.6081460120615665, 0.41085311660812],
# [0.6687592319054653, 0.5397741327692814, 0.6501861150662493, 0.4353542790743682],
# [0.6698670605612999, 0.537673090495975, 0.6501070593469739, 0.4258987360492556],
# [0.6661742983751846, 0.5296012096868737, 0.6368143929320171, 0.4275678587834551],
# [0.6676514032496307, 0.5308785119180587, 0.636242212888582, 0.4264897717169535]]

data = np.array(data)
acc = data[:,0]
nmi = data[:,1]
f1 = data[:,2]
acc = acc.reshape((4,4))
nmi = nmi.reshape((4,4))
f1 = f1.reshape((4,4))


acc = pd.DataFrame(acc, index=[0.9, 0.7, 0.5, 0.2], columns=[0.2, 0.5, 0.7, 0.9])
nmi = pd.DataFrame(nmi, index=[0.9, 0.7, 0.5, 0.2], columns=[0.2, 0.5, 0.7, 0.9])
f1 = pd.DataFrame(f1, index=[0.9, 0.7, 0.5, 0.2], columns=[0.2, 0.5, 0.7, 0.9])

plt.figure(dpi=120)
sns.heatmap(data=acc, fmt='d', linewidths=.5, cmap='YlGnBu')
plt.show()