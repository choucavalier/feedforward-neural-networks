from linear_regression import *
from numpy import matrix, random, transpose

# Total error: sum of the errors over the training set.
def error_tot(t, w):
    return sum([error(x, y, w) for x, y in t])

# Arbitrary target function with two inputs.
def f(x0, x1):
    return 5 + 2 * x0 - 3 * x1

# 100 random pair of input (matrix of size 1, 2) between 0 and 1
x = [[random.rand(), random.rand()] for i in range(100)]
y = [f(xi[0], xi[1]) for xi in x]

# Add add bias and use matrix instead of lists.
x = [matrix([1] + xi) for xi in x]

# Training set is a pair of (input, target).
# zip takes the pair of list and transforms it into a list of pairs.
t = list(zip(x, y))

# Start with a column matrix of random weights.
# t[0] is the first example, t[0][0] is the input of the first example.
# So t[0][0].size is the number of inputs.
w = random.rand(t[0][0].size, 1)

print(error_tot(t, w))
for i in range(1, 201):
    # For each example in the training set, learn from the example. Repeat
    # as necessary.
    for x, y in t:
        w = learn(x, y, w, 0.005)
    print(error_tot(t, w))
print()


# Should be close to the coefficients in f: [5 2 -3]
print(w)
