from multilayer_perceptron import *
from numpy import matrix, multiply, random, log

# Error function of one neuron.
# o: output of the model.
# y: target.
def error(o, y):
    return multiply(-y, log(o)) - multiply((1 - y), log(1 - o))

# Error for one training example = sum of the errors for each output.
def error_sum(o, y):
    return sum([error(o[j], y[j]) for j in range(len(o))])

# Error function. We use it to monitor the learning process and make sure the
# model converges.
def error_tot(t, w):
    return sum([error_sum(h(x, w)[-1], y) for x, y in t])

# Training set: a list of pair (input, target).
t = [
    ([0, 0], [1, 1]),
    ([0, 1], [0, 0]),
    ([1, 0], [0, 1]),
    ([1, 1], [1, 0])]

# Use column matrices instead of lists.
t = [(transpose(matrix(x)), transpose(matrix(y))) for x, y in t]

# Start with two random weights matrices.
w = [random.rand(3, 3), random.rand(4, 2)]

print(error_tot(t, w))
for i in range(1, 1001):
    for x, y in t:
        w = learn(x, y, w, 0.3)
    print(error_tot(t, w))

print()
# For each input, show what the model has learnt.
for x, y in t:
    print('h(', x, ') = ', h(x, w)[-1])
