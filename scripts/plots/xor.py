import numpy as np
import matplotlib.pyplot as plt

font = { 'family': 'normal', 'weight': 'bold', 'size': 30 }
plt.rc('font', **font)

x = [0, 1, 1, 0]
y = [1, 0, 1, 0]
colors = ['#000000', '#000000', '#ffffff', '#ffffff']
areas = [170 for i in range(len(x))]
plt.scatter(x, y, s=areas, c=colors, alpha=1.0)
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.yticks(np.arange(min(y), max(y)+1, 1.0))

plt.xlim([-0.2, 1.2])
plt.ylim([-0.2, 1.2])

plt.title('XOR: Not Linearly Separable')
plt.text(0.5, 0.5, r'??', ha='center', va='center', size='x-large')

plt.savefig('xor.png')
