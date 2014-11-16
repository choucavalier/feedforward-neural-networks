import numpy as np
import matplotlib.pyplot as plt

x = [0, 1, 1, 0]
y = [1, 0, 1, 0]
colors = ['#000000', '#000000', '#ffffff', '#ffffff']
areas = [170 for i in range(len(x))]
plt.scatter(x, y, s=areas, c=colors, alpha=1.0)
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.yticks(np.arange(min(y), max(y)+1, 1.0))
plt.margins(0.2)
plt.subplots_adjust(bottom=0.15)
plt.title('XOR: Not Linearly Separable')
plt.text(0.5, 0.5, r'??', ha='center', va='center', size='x-large')
plt.show()
