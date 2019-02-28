import numpy as np
from path_design.create_path import ReferencePath
from config.parameter_setting import MPCConfig
import matplotlib.pyplot as plt
import matplotlib.animation as animation

path=ReferencePath(MPCConfig()).ref_path
data = np.loadtxt('out.csv', delimiter=',')
fig = plt.figure()

ims = []
for i in range(1000):
    im = plt.scatter(data[i*15, 0], data[i*15, 1], color='blue')
    ims.append([im])

# アニメーション作成
ani = animation.ArtistAnimation(fig, ims, interval=1)
plt.plot(path[:, 0], path[:, 1], color='red')
ani.save('anim.gif', writer="pillow")

# 表示
plt.show()
