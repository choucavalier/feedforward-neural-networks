from logistic_regression import *
from numpy import matrix, random
from matplotlib import pyplot as plt

# Plot the decision boundary.
def plot_db(t, w, i):
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    ax.axis([0, 10, 0, 10])
    ax.set_xlabel('x0')
    ax.set_ylabel('x1')
    ax.autoscale(False)

    # With weights [w0 w1 w2], the decision boundary is the line where:
    #   w0 + w1 * x1 + w2 * x2 == 0
    # By isolating x2, we get the equation of the decision boundary in the plane:
    #   x2 = (-w0 - w1 * x1) / w2
    ax.plot([0, 10], [-w[0, 0] / w[2, 0], (-w[0, 0] - w[1, 0] * 10) / w[2, 0]], color='r')
    for x, y in t:
        if y == 0:
            ax.scatter(x[0, 1], x[0, 2], marker='x', s=100., color='g')
        else:
            ax.scatter(x[0, 1], x[0, 2], marker='o', s=100.)
    print('Saving logreg_db{0:03d}.png...'.format(i))
    fig.savefig('logreg_db{0:03d}.png'.format(i))
    plt.close(fig)

# Training set: a list of pair (input, target).
t = []
for i in range(100):
    # Input is a random pair of numbers between 0 and 10.
    x = [random.random() * 10, random.random() * 10]

    # Some arbitrary linearly separable function.
    y = 1 if 2 * x[0] + x[1] - 12 > 0 else 0

    t.append((x, y))

# Add bias and use matrix instead of lists for inputs.
t = [(matrix([1] + x), y) for x, y in t]

# Start with random weights. This is a column matrix with as many rows as
# inputs.
w = random.rand(t[0][0].size, 1)

plot_db(t, w, 0)
for i in range(1, 100):
    for x, y in t:
        w = learn(x, y, w, 1)
    plot_db(t, w, i)
