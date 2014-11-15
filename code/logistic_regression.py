from numpy import exp, transpose

# Logistic function.
def g(x):
    return 1 / (1 + exp(-x))

# Hypthesis: the prediction of our model for input x when parametrized with
# weights w.
def h(x, w):
    return g(transpose(w) * x)

# Gradient of the error function with respect to the weights w.
def gradient(x, y, w):
    # The gradient of the error with respect to a weight wi is:
    #   dE/dwi = x[i] * (o - y)
    # where o is the output of the model, h(x, w).
    #
    # We want to compute this value for each i:
    #
    #          dE/w00
    #           ... 
    #          dE/wi0
    # 
    #             =
    #
    #      x[0] * (o - y)
    #            ...      
    #      x[i] * (o - y)
    #
    # which is the multiplication of x with (o - y).
    return x * (h(x, w) - y)

# Adjust the weights using gradient descent.
def learn(x, y, w, a):
    return w - a * gradient(x, y, w)
