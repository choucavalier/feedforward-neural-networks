from numpy import matrix, transpose

# Error function. We use it to monitor the learning process and make sure the
# model converges.
def error(x, y, w):
    return (h(x, w) - y) ** 2

# Hypthesis: the prediction of our model for input x when parametrized with
# weights w.
def h(x, w):
    return x * w

# Gradient of the error function with respect to the weights w.
def gradient(x, y, w):
    # The gradient of the error with respect to a weights wi is:
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
    # which is the multiplication of the transpose of x with (o - y).
    return transpose(x) * (h(x, w) - y)

# Adjust the weights using gradient descent.
def learn(x, y, w, a):
    return w - a * gradient(x, y, w)
