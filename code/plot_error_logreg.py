from logistic_regression import *
from numpy import matrix, multiply, random, log, meshgrid, zeros
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Error function of one neuron.
# o: output of the model.
# y: target.
def error(o, y):
    return multiply(-y, log(o)) - multiply((1 - y), log(1 - o))

def error_tot(t, w):
    return sum([error(h(x, w), y) for x, y in t])

# Plot the error as a 3D surface.
# Only defined for 2 weights
def plot_error(t, ax):
    w0 = list(range(-15, 16))
    w1 = list(range(-15, 16))
    W0, W1 = meshgrid(w0, w1)
    Z = zeros((len(w1), len(w0)))
    for j in range(len(w1)):
        for i in range(len(w0)):
            w = transpose(matrix([w0[i], w1[j]]))
            Z[j, i] = error_tot(t, w)
    return W0, W1, Z

def plot_learning_path(t, ax):
    # Start with some arbitrary weights.
    w = transpose(matrix([-10, -10]))

    pathx = [w[0, 0]]
    pathy = [w[1, 0]]
    pathz = [error_tot(t, w)[0, 0]]
    for i in range(1, 31):
        for x, y in t:
            w = learn(x, y, w, 0.3)
        pathx.append(w[0, 0])
        pathy.append(w[1, 0])
        pathz.append(error_tot(t, w)[0, 0])
    return pathx, pathy, pathz

# Training set: a list of pair (input, target).
t = []
for i in range(10):
    # Input is a random number between -0.5 and 0.5.
    x = [random.random() - 0.5]

    # Some arbitrary target function.
    y = 1 if 20 * x[0] ** 2 - 10 * x[0] < 0 else 0

    t.append((x, y))

# Add bias and use column matrices instead of lists for inputs.
t = [(transpose(matrix([1] + x)), y) for x, y in t]

# Start with random weights. t[0][0].size is the number of inputs.
w = random.rand(t[0][0].size, 1)

fig1 = plt.figure(0)
ax = fig1.add_subplot(111, projection='3d')
ax.set_xlabel('w0')
ax.set_ylabel('w1')
ax.set_zlabel('error')
W0, W1, Z = plot_error(t, ax)
ax.plot_surface(W0, W1, Z, cstride=1, rstride=1, cmap=plt.cm.jet, alpha = 0.5)
ax.view_init(30, 60)
plt.tight_layout()

fig2 = plt.figure(1)
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot_surface(W0, W1, Z, cstride=1, rstride=1, cmap=plt.cm.jet, alpha = 0.5)
ax2.set_xlabel('w0')
ax2.set_ylabel('w1')
ax2.set_zlabel('error')
pathx, pathy, pathz = plot_learning_path(t, ax2)
ax2.plot(pathx, pathy, pathz, color='r', marker='o')
ax2.view_init(30, 60)
plt.tight_layout()
plt.show(block=True)
