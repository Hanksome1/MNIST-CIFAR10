import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0.0, 2*np.pi)  # 50x1 array between 0 and 2*pi
y = np.cos(x)                  # cos(x)

plt.plot(x, y, 'r')   # red line without marker
plt.savefig("relu.png")
