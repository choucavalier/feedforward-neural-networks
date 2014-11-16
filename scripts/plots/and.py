import numpy as np
import matplotlib.pyplot as plt

x = [0, 0, 1, 1]
y = [0, 1, 0, 1]
colors = ['#ffffff', '#ffffff', '#ffffff', '#000000']
areas = [170 for i in range(len(x))]
plt.scatter(x, y, s=areas, c=colors, alpha=1.0)
plt.xticks(np.arange(min(x), max(x) + 1, 1.0))
plt.yticks(np.arange(min(y), max(y) + 1, 1.0))
plt.title('AND')

plt.xlim([-0.2, 1.2])
plt.ylim([-0.2, 1.2])

linex = [0.4, 1.2]
liney = [1.2, 0.4]

plt.plot(linex, liney)

plt.show()
