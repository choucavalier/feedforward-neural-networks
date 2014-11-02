from single_layer_perceptron import *
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
    return sum([error_sum(h(x, w), y) for x, y in t])

# Training set: a list of pair (input, target).
t = [
    ([0, 0], [0, 0]),
    ([0, 1], [0, 1]),
    ([1, 0], [0, 1]),
    ([1, 1], [1, 1])]

# Add bias and use matrix instead of lists.
t = [(matrix([1] + x), matrix(y)) for x, y in t]

# Start with random weights. There are as many rows as inputs and as many
# columns as outputs.
w = random.rand(t[0][0].size, t[0][1].size)

print(error_tot(t, w))
for i in range(1, 101):
    for x, y in t:
        w = learn(x, y, w, 1)
    print(error_tot(t, w))

print()
# For each input, show what the model has learnt.
for x, y in t:
    print('h(', x, ') = ', h(x, w))
