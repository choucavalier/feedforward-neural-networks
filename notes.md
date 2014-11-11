# 01-introduction

* What is Machine Learning?
  + Use experimental data to create a statistical model capable of predicting
  unknown data

* Example of a Stastical Model
  * IQ at EPITA, normal distrubition

# Binary classification


* h(x) = g(x.w)
  h: 0 if x < 0
     1 if x > 0

* Express the result as a probability using logistic function (sigmoid)

# Logistic regression

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

# Single layer perceptron

* what if we want multiple classes? We add neurons

[PROGRESSIVE DIAGRAM]
adding classes (and then neurons and links)

# Artificial neural network

* diagram of a non linearly-separable data set
* we could use polynomials of higher degrees to represent our hypothesis
function -> and then leave the bisounours linear world
* what tells you that your problem is polynomialy separable?


[BREAK]
Make sure everyone is OK with what we talked about

* Universal approximation theorem: for any continuous function, there is a
feed-forward neural network that computes it.

[PROGRESSIVE DIAGRAM]
adding a hidden layer

# Backpropagation algorithm

# Optimizations

* momentum
* overfitting
* regularization
