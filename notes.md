# 01-introduction

* What is Machine Learning?
  + Use experimental data to create a statistical model capable of predicting
  unknown data

* Example of a Stastical Model
  * IQ at EPITA, normal distrubition

# 02-linear-regression

## Example: price of houses depending on m²

* Establishing the mathematical model
* Math notation: matrix multiplication
* Training set
* Error

[PROGRESSIVE DIAGRAM]
linear combination
show multiple inputs (e.g. number of rooms)

## Gradient descent

* Convergence

[LIVE CODING]
Linear regression on a set of data. matplotlib

## Multi-variable linear regression

* We can extend the concept to multiple variables by increasing the number of
weights

# 03-logistic-regression

* Sometimes we want a binary output rather than finding out a continuous
variable
* Example: lung cancer
* h(x) = g(x.w)
  h: 0 if x < 0
     1 if x > 0
* We have 2 classes!
* Express the result as a probability using logistic function (sigmoid)
* Recall: training set, what is our training set here?
* New error formula, same as for ANN
* Logistic regression
* Concept of 'decision boundary'
  * diagram circles and crosses to show the classes
  * show that the data is linearly separable
    -> we can draw a line that separates data, that's what it means

[PROGRESSIVE DIAGRAM]
sigmoid function
show pixels as input

# 04-single-layer-perceptron

* what if we want multiple classes? We add neurons

[PROGRESSIVE DIAGRAM]
adding classes (and then neurons and links)

# 05-non-linearly-separable-data

* diagram of a non linearly-separable data set
* we could use polynomials of higher degrees to represent our hypothesis
function -> and then leave the bisounours linear world
* what tells you that your problem is polynomialy separable?

# 06-artificial-neural-network

[BREAK]
Make sure everyone is OK with what we talked about

* Universal approximation theorem: any neural network can represent any
continuous function

[PROGRESSIVE DIAGRAM]
adding a hidden layer

# 07-backpropagation-algorithm

# 08-optimizations

* momentum
* overfitting
* regularization
