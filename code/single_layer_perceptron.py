from numpy import exp, transpose

# Logistic function.
# N.B. The function is defined for vectors.
# g([x0 ... xn]) = [g(x0) ... g(xn)]
def g(x):
    return 1 / (1 + exp(-x))

# Hypthesis: the prediction of our model for input x when parametrized with
# weights w.
def h(x, w):
    return g(transpose(w) * x)

# Gradient of the error function with respect to the weights w.
def gradient(x, y, w):
    # The gradient of the error with respect to a weights wij is:
    #   dE/dwij = x[i] * (o[j] - y[j])
    # where o is the output of the model, h(x, w).
    #
    # This can be rewritten as
    #   dE/dwij = x[i] * (o - y)[j]
    # since the '-' operator is defined element wise for vectors.
    #
    # We want to compute this value for each i and each j:
    #
    #                 dE/w00  ...  dE/w0j
    #                   ...   ...   ...
    #                 dE/wi0  ...  dE/wij
    #
    #                          =
    #
    #      x[0] * (o - y)[0]  ...  x[0] * (o - y)[j] 
    #              ...        ...        ...         
    #      x[i] * (o - y)[0]  ...  x[i] * (o - y)[j] 
    #
    # which is the multiplication of x with the transpose of (o - y).
    return x * transpose((h(x, w) - y))

# Adjust the weights using gradient descent.
def learn(x, y, w, a):
    return w - a * gradient(x, y, w)
