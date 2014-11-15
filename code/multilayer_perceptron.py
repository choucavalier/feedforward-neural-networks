from numpy import matrix, multiply, exp, transpose, append

# Logistic function.
# N.B. The function is defined for vectors.
# g([x0 ... xn]) = [g(x0) ... g(xn)]
def g(x):
    return 1 / (1 + exp(-x))

# Hypothesis: the prediction of our model for input x when parametrized with
# weights w.
def h(x, w):
    # out will contain the output of each layer.
    # The first layer is the input x.
    out = [x]

    for l in range(len(w)):
        # Add bias to previous layer's output
        x = append(matrix([1]), out[l], axis=0)

        # Append output to the list of outputs.
        out.append(g(transpose(w[l]) * x))

    # Note that out has one more element than w, which is perfectly normal:
    #           w0             w1
    #           |              |
    #   out0 --g(*) -> out1 --g(*)-> out2
    return out

# Gradient of the error function with respect to the weights w.
def gradient(out, y, w):
    # We need to compute the gradient of the last layer L, and then use it to
    # retropropagate the error's gradient through the layers up until the first
    # one.
    L = len(out) - 1

    # Input of last layer = output of previous layer + bias.
    x = append(matrix([1]), out[L - 1], axis=0)

    # Gradient is the same as for logistic regression, but we store the value
    # out[L] - y because we need it to compute the previous layer's gradient.
    # (it corresponds to the derivative of the error with respect to net[L])
    d = out[L] - y
    grad = [x * transpose(d)]

    # In python, if l is a list, l[1:-1] is the same list without the first
    # and last element. We already processed the last element (the output
    # layer) and the first layer is just the input layer so we remove them.
    # We want to go through the layers in reverse order. That's what [::-1] do:
    # it reverses the list.
    for l in range(len(out))[1:-1][::-1]:
        
        # Derivative of the error with respect to current layer's output.
        d_out = w[l][1:] * d

        # Derivative of the error with respect to net[l]
        d = multiply(d_out, multiply(out[l], 1 - out[l]))

        # Current layer's input = previous layer's output + bias.
        x = append(matrix([1]), out[l - 1], axis=0)

        # Derivative of the error with respect to the current layer's weights.
        grad = [x * transpose(d)] + grad
    return grad

# Adjust the weights using gradient descent.
def learn(x, y, w, a):
    out = h(x, w)
    L = len(out) - 1
    grad = gradient(out, y, w)

    # Update each layer.
    for l in range(len(w)):
        w[l] = w[l] - a * grad[l]
    return w
