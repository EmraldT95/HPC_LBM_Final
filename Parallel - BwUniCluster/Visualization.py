import numpy as np
import matplotlib.pyplot as plt

ux_kl = np.load('ux.npy')
uy_kl = np.load('uy.npy')

nx, ny = ux_kl.shape
fig, ax = plt.subplots()
x_k = np.arange(nx-2)
y_l = np.arange(ny-2)
ax.set(title='Omega=1.7, Wall velocity (TOP) = 0.1',xlabel='X-direction',ylabel='Y-direction')
ax.streamplot(x_k, y_l, ux_kl[1:-1, 1:-1].T, uy_kl[1:-1, 1:-1].T, density=3)
plt.show()