# Machine Learning
Here are some notes I took from doing courses/reading books.
- [Machine Learning - Andrew N.](https://www.coursera.org/learn/machine-learning)

## Contents
- [What is Machine Learning?](#what-is-machine-learning)
- [Supervised Learning](#supervised-learning)
- [Unsupervised Learning](#unsupervised-learning)
    - [Clustering Algorithm](#clustering-algorithm)
    - [Cocktail Party Algorithm](#cocktail-party-algorithm)
- [Cost Function](#cost-function)
    - [Formula](#formula)
- [Gradient Descent](#gradient-descent)
    - [Formula](#formula-1)
    - [Learning Rate](#learning-rate-alpha)
    - ["Batch" Gradient Descent](#batch-gradient-descent)
    - [Debugging](#debugging)
- [Matrix](#matrix)
    - [Elements](#elements)
    - [Addition](#addition)
    - [Scalar Multiplication/Division](#scalar-multiplicationdivision)
    - [Multiplication with a Vector](#multiplication-with-a-vector)
    - [Matrix Matrix Multiplication](#matrix-matrix-multiplication)
    - [Identity Matrix](#identity-matrix)
    - [Inverse](#inverse)
    - [Transpose](#transpose)
    - [Properties](#properties)
    - [MATLAB](#matlab)
- [Vector](#vector)
    - [Elements](#elements-1)
    - [MATLAB](#matlab-1)
- [Linear Regression with Multiple Features](#linear-regression-with-multiple-variables-features)
    - [Formula](#formula-2)
- [Gradient Descent for Multiple Features](#gradient-descent-for-multiple-variables)
    - [Feature Scaling](#feature-scaling)
    - [Mean Normalization](#mean-normalization)
- [Polynomial Regression](#polynomial-regression)
- [Normal Equation](#normal-equation)
    - [Dimensions](#dimensions)
    - [Non-Invertibility](#non-invertibility)
       - [Causes](#causes)
    - [MATLAB](#matlab-2)
    - [Example](#example)
    - [Gradient Descent vs Normal Equation](#gradient-descent-vs-normal-equation)
- [Classification and Representation](#classification-and-representation)
    - [Hypothesis Representation](#hypothesis-representation)
       - [Formula](#formula-3)
       - [Sigmoid Function](#sigmoid-function)
          - [MATLAB](#matlab-3)
       - [Decision Boundary](#decision-boundary)
          - [Example](#example-1)
          - [MATLAB](#matlab-4)
- [Logistic Regression Model](#logistic-regression-model)
    - [Cost Function for Logistic Regression](#cost-function-for-logistic-regression)
    - [Simplified Cost Function](#simplified-cost-function)
    - [Simplified Gradient Descent](#simplified-gradient-descent)
    - [Advanced Optimization](#advanced-algorithms)
       - [Optimization Algorithms](#optimization-algorithms)
       - [MATLAB](#matlab-5)
- [Multiclass Classification](#multiclass-classification)
    - [One-vs-all](#one-vs-all)
- [The Problem of Overfitting](#the-problem-of-overfitting)
    - [Cost Function](#cost-function-1)
    - [Regularized Linear Regression](#regularized-linear-regression)
       - [Gradient Descent](#gradient-descent-1)
       - [Normal Equation](#normal-equation-1)
    - [Regularized Logistic Regression](#regularized-logistic-regression)
       - [Cost Function](#cost-function-2)
          - [MATLAB](#matlab-6)
- [Motivations](#motivations)
    - [Non-linear Hypothesis](#non-linear-hypothesis)
    - [Neurons and the Brain](#neurons-and-the-brain)
       - [The "one learning algorithm" hypothesis](#the-one-learning-algorithm-hypothesis)
- [Neural Networks](#neural-networks)
    - [Model Representation 1](#model-representation-1)
       - [Example](#example-2)
    - [Model Representation 2](#model-representation-2)
    - [Examples and Intuitions](#examples-and-intuitions)
       - [One](#one)
       - [Two](#two)
       - [Multiclass Classification](#multiclass-classification-1)

## What is Machine Learning?
> "A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E." - Tom Mitchell

e.g. playing checkers  
E = experience of playing many games  
T = task of playing checkers  
P = probability that the program will win the next game

## Supervised learning
We are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and the output.

 * Right answers given
 * Regression (goal is to predict a continuous valued output)
 * Classification (output 0 or 1)

Data set is called **Training Set**. It is then being fed to our learning algorithm. Will output a function, which by convention is usually denoted *h*.  
h stands for hypothesis > the function maps from x's to y's  
```
hθ(x) = θ0 + θ1x
h(x)
```

## Unsupervised learning
We are given data, but don't know what to do with it. Let the algorithm figure it out.

### Clustering algorithm

 * Given a set of news articles, automatically group them into sets of articles about the same story
 * Given a database of customer data, discover market segments and group customers into different market segments

### Cocktail party algorithm
find structure in a chatoic environment

```
[W,s,v] = svd((repmat(sum(x.*x, 1), size(x, 1), 1).*x)*x^i);
```
 * svd = single value decomposition
 * repmat = replicate and tile an array
 
## Cost function
We can measure the accuracy of our hypothesis by using the cost function. It takes an average difference of all the results.  
Idea is to choose θ0 and θ1 so that hθ(x) is close to our data examples (x, y).  

### Formula
<img width="40%" height="40%" src="https://github.com/Pasoy/ml-projects/blob/master/images/cost_function.png">    
Calculation is the hypothesis value for h(x), minus the actual value of y and then we square

```python
import numpy as np

X = np.array([[1], [2], [3]])
y = np.array([[0.5], [1], [1.5]])

get_theta = lambda theta: np.array([[0, theta]])

thetas = list(map(get_theta, [0.5, 1.0, 1.5]))

X = np.hstack([np.ones([3, 1]), X])

def cost(X, y, theta):
    inner = np.power(((X @ theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

for i in range(len(thetas)):
    print(cost(X, y, thetas[i]))
```  

e.g  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/eg_cf_graph1.png">  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/eg_cost_function.png">

## Gradient descent
Algorithm which is the foundation of many others.  
*We want to find the lowest point*.  
The *weights* need to be adjusted as if we are "going down". We make small adjustments to our weights that we are slowly getting closer to the lowest point.  
We can calculate the derivative of our function to see which way is going downhill.  
For functions used in linear regression, there is only a global optimum (no local optima).  

### Formula 
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/gradient_descent.png">  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/gradient_descent_1.png">  

 * Alpha is the learning rate (controls how big a step is)
 * Alpha is always a positive number
 * Term after alpha is called the derivative
 * := is an assignment
 * j represents the feature index number  

In gradient descent we simultaneously update θ0 and θ1.  
If already at the local minimum it will not change.

### Learning rate (alpha)
 * Is it is too small, gradient descent can be slow
 * If it is too large, gradient descent can overshoot the minimum. it may fail to converge or even diverge
 * After descent, it will automatically take smaller steps
 * To choose the rate, try a range of numbers like `..., 0.001, 0.01, 0.1, 1,..`
 
### "Batch" gradient descent
Means each step of gradient descent uses all training data.

### Debugging
Make a plot with *number of iterations* (noi) on the x-axis. plot the cost function over the *noi* of gradient descent.  
If the cost function increases, we probably need to decrease our learning rate. 

***

# Linear algebra

## Matrix
 * Is an rectangular array of numbers.  
 * Dimension of matrix: number of rows * number of columns.  
 * Usually uppercase
 
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/matrix_example_1.png">  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/matrix_example_2.png">  

### Elements
```
A(i,j) = "i, j entry" in the row i and in the column j
e.g. from first matrix example
A(2,1) = 3
```

```matlab
% The ; refers to a new line
A = [1, 2, 3; 4, 5, 6; 7, 8, 9; 10, 11, 12]

% Get dimension of the matrix
[m,n] = size(A)

% You could also store it this way
dim_A = size(A)

% Now let's index into the 2nd row 3rd column of matrix A
A_23 = A(2,3)

```

### Addition
To add matrices, they have to be the same dimension.  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/matrix_addition.png"> 

### Scalar multiplication/division
*Scalar is a real number*  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/matrix_scalar.png">  

### Multiplication with a vector
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/matrix_vector_multi.png">  

 * Result is a vector
 * Number of columns of the matrix must equal the number of rows of the vector

Here is a website for good visualization: http://matrixmultiplication.xyz/

### Matrix matrix multiplication
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/matrix_matrix_multi.png">  

 * Result is a matrix (dimension: M1 columns * M2 rows)
 * Number of columns in the first matrix must match the number of rows in the second matrix
 * Can be used to predict something (if `h(x)` is given and sample data; see below)
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/matrix_matrix_multi_predictions.png">

### Identity matrix
 * Denoted `I (or In*n)`
 * It can be a different dimension
 * For any matrix A, `A*I = I*A = A`
 
### Inverse
 * If a matrix is a square matrix (m*m), and has an inverse, then `A*(A^(-1)) = I`. I being the identity matrix
 * Matrices which don't have an inverse are "**singular**" or "**degenerate**"
 
### Transpose
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/matrix_transpose_1.png">  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/matrix_transpose_2.png">  

 * Means sort of flipping the matrix
 * Denoted `A *superscript* T`
 * Rows become the columns
 * Columns become rows
 * `B(i,j) = A(j,i)`

### Properties
 * `A * B != B * A` - not commutitive
 * `A*(B*C) = (A*B)*C` - is associative

### MATLAB
```matlab
% Create matrices
A = [1, 2, 4; 5, 3, 2]
B = [1, 3, 4; 1, 1, 1]

% Create a 3 by 3 identity matrix
I = eye(3)

% The above notation is the same as I = [1,0,0;0,1,0;0,0,1]

% Initialize constant s 
s = 2

% See how element-wise addition works
add_AB = A + B 

% See how element-wise subtraction works
sub_AB = A - B

% See how scalar multiplication works
mult_As = A * s

% Divide A by s
div_As = A / s
```

***

## Vector
 * Usually lowercase
 * Is an `n * 1` matrix
 * Dimension = number of rows
 * 1-indexed (more common) / 0-indexed vectors
 
### Elements
```
y(i) = the element if row i
```

### MATLAB
```matlab
% Create a vector 
v = [1;2;3] 

% Get the dimension
dim_v = size(v)
```

***

## Linear regression with multiple variables (features)
 * n = number of vars
 * x(i) = input of i-th training example
 * xj(i) = value of var j in i-th training example

We can improve the features and form the hypthesis in different ways. We can **combine** multiple features into one. (`x1 and x2 -> x3 = x1*x2`)

### Formula
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/reg_new_hypothesis.png">  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/reg_new_hypothesis_1.png">  

## Gradient descent for multiple variables
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/gradient_descent_new.png">  

 * Need to choose learning rate
 * Needs many iterations
 * Works well even when features are large (if larger than e.g. *10^6*)

### Feature scaling
The idea is that the features are on a similar scale.   
 * It takes less time, because it will descend quickly on small ranges
   * Speeds up gradient descent by making it require fewer iterations to get to a good solution
 * More accurate
 * Less complicated  

get every feature into a `-1 =< x(i) <= 1` range. of course, it can be a bit bigger.

### Mean normalization
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/gradient_descent_mean.png">  

Replace `x(i) with x(i) - μ(i)`. μ being the average value of the training examples. also divide by `s(i)` (the range of values (max-min) ).  
```
e.g we have an algorithm to estimate the life span of a tree.
in our training set, we have trees with the life span between 5 and 30 years. our average is 13 years.
x(i) = (life span - 13) / 25
```

## Polynomial regression
It is possible to change the curve of our hypothesis by making it *quadratic*, *cubic* or to the *square root* and more.  
It is **important** to scale our features! Because `x1 = 100 -> x1^2 = 10,000 -> x1^3 ? 1,000,000`

```
e.g h(x) = θ0 + θ1 * x1
we can create additional features based on our x1
- quadratic: h(x) = θ0 + θ1 * x1 + θ2 * x1^2
- cubic: h(x) = θ0 + θ1 * x1 + θ2 * x1^2 + θ3 * x1^3
- square root: h(x) = θ0 + θ1 * x1 + θ2 * sqrt(x1)
```

## Normal equation
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/normal_equation.png">  
a method to solve for θ analytically.  

 * No need to choose learning rate
 * Do not need to iterate
 * Slow if features are very large
 * Works well if small features (e.g. *10^2*)
 
### Dimensions
 * `X` has m rows and n + 1 columns (+1 because of the x0=1)  
 * `y` is an m-vector.  
 * `θ` is an (n+1)-vector.  

### Non-invertibility
#### Causes
 * Redundant features (linearly dependent)
   * `x1 = size in feet^2`
   * `x2 = size in m^2`
 * Too many features
   * delete some features, or use regularisation

### MATLAB
```matlab
pinv(transpose(X) * X) * transpose(X) * y
```

### Example
1. we need to create a new feature an fill it with 1s.  
2. convert to matrices and vectors.  
<img width="50%" height="50%" src="https://github.com/Pasoy/ml-projects/blob/master/images/normal_equation_ex1.png">  
<img width="50%" height="50%" src="https://github.com/Pasoy/ml-projects/blob/master/images/normal_equation_ex1_done.png">  
3. After that we need to fill it in, in our formula.  
 
### Gradient descent vs Normal equation

Gradient Descent | Normal Equation
------------ | -------------
needs to choose alpha | no need to choose alpha
many iterations | Content in the second column
works well when features large | slow if features large

# Classification and Representation
 * Email: Spam / Not Spam?
 * News: Fake (Yes / No)
  
This is a binary classification problem. We are trying to predict a value `y` and have a result of either `1` or `0`.
 * 0: "Negative Class"
 * 1: "Positive Class"
  
Applying linear regression to a classification problem is not a great idea. Cost function may change if one data sample is far away from the other training examples.

## Hypothesis Representation
We want `0 < h(x) < 1`.

### Formula
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/hypothesis_representation.png">  

The new formula uses the **Sigmoid Function**, also known as **Logistic Function**.  

`h(x)` will give us the **probability** that our output is 1.
```matlab
h(x) = 0.7 % gives us a probability of 70% that our output is 1
           % the probability of 0 is just the complement of our probability that is is 1
           % 1: 70% 0: 30%
```
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/hypothesis_probability.png">  

### Sigmoid Function
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/sigmoid_function.png">  

#### MATLAB
```matlab
function g = sigmoid(z)

g = zeroes(size(z));

g = 1 ./ (1 + exp(-z));

end
```

### Decision Boundary
The **decision boundary** is the line that separates the area where `y = 0` and `y = 1`. It is created by our hypothesis function.  

In order to get our discrete 0 or 1 classification, we can translate the output of `h(x)` as:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/decision_1.png">  

The way the logistic function `g` behaves is that when its input is greater than or equal to zero, its output is greater than or equal to 0.5  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/decision_2.png">  

That means:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/decision_3.png">  

From the previous statements we can now say:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/decision_4.png">  

#### Example
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/decision_5.png">  

In this case, our decision boundary is a straight vertical line placed on the graph where `x1 = 5`, and everything to the left of that denotes`y = 1`, while everything to the rightt denotes `y = 0`.  

#### MATLAB
```matlab
% predict whether the label is 0 or 1 using learned logistic regression parameters theta
% like if it is bigger than or equal to 0.5, predict 1

m = size(X, 1); % number of training examples

p = zeroes(m, 1);

hypothesis = sigmoid(X * theta);
p = (hypothesis >= 0.5);

end
```

# Logistic Regression Model

## Cost Function for Logistic Regression
We cannot use the same cost function that we use for linear regression because the Logistic Function will cause the output to be wavy, causing many local optima. In other words, it will not be a convex function.  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/logistic_regression_1.png">  

When `y = 1`, we get the following plot for `J` vs `h(x)`:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/logistic_regression_2.png">  

When `y = 0`, we get the following plot for `J` vs `h(x)`:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/logistic_regression_3.png">  

<img src="https://github.com/Pasoy/ml-projects/blob/master/images/logistic_regression_4.png">  

If our correct answer `y is 0`, then the cost function will be 0 if our hypothesis function outputs 0. If it approaches 1, then the cost function will approach infinity.  
If our correct answer `y is 1`, then the cost function will be 0 if our hypothesis function outputs 1. If it approaches 0, then the cost function will approach infinity.  

Note that writing the cost function this way guarantees that `J` is convex.

## Simplified Cost Function
We can compress the cost function's two conditional cases into one:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/simple_cost_function_1.png">  

So we can fully write the entire function as follows:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/simple_cost_function_2.png">  

A vectorized implementation is:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/simple_cost_function_3.png">  

## Simplified Gradient Descent
The general form of gradient descent is:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/gradient_descent.png">  

We can work out the derivative part using calculus to get:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/simple_gd_1.png">  

Notice that this algorithm is identical to the one we used in linear regression. We still have to simultaneously update all values in theta.  
A vectorized implementation is:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/simple_gd_2.png">  

## Advanced Optimization

### Optimization algorithms
 * Gradient descent
 * Conjugate gradient
 * BFGS
 * L-BFGS
 
"Conjugate gradient", "BFGS", and "L-BFGS" are more sophisticated, faster ways to optimize θ that can be used instead of gradient descent.  

The advantages of the last 3 algorithms:
 * No need to manually pick alpha
 * Often faster than gradient descent
Disadvantages:
 * More complex
 
### MATLAB
```matlab
% a cost function - example where theta1 = 5 and theta2 = 5
function [jVal, gradient] = costFunction(theta)
    jVal = (theta(1) - 5) ^ 2 + 
           (theta(2) - 5) ^ 2; % code to compute J(theta)
    gradient = zeroes(2,1); % create vector
    gradient(1) = 2 * (theta(1) - 5); % derivative for theta 0
    gradient(2) = 2 * (theta(2) - 5); % derivative for theta 1

% optimization
options = optimset('GradObj', 'on' 'MaxIter', 100);
initialTheta = zeroes(2,1);
[optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options)
```

# Multiclass Classification
e.g.  
 * Email foldering: Work `(y = 1)`, School `(y = 2)`, Hobby `(y = 3)`, Family `(y = 4)`
 * Medical diagrams: Not ill `(y = 1)`, Cold `(y = 2)`, Flu `(y = 3)`
 
## One-vs-all
Now we will approach the classification of data when we have more than two categories. Instead of y = {0,1} we will expand our definition so that y = {0,1...n}.  

Since y = {0,1...n}, we divide our problem into n+1 (+1 because the index starts at 0) binary classification problems; in each one, we predict the probability that 'y' is a member of one of our classes.  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/onevsall.png">  

We are basically choosing one class and then lumping all the others into a single second class. We do this repeatedly, applying binary logistic regression to each case, and then use the hypothesis that returned the highest value as our prediction.  

# The Problem of Overfitting
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/overfitting.png">  

The left figure shows the result of fitting a `y = theta(0) + theta(1)x` to a dataset. The fit is not very good.  
Instead, if we added an extra feature x^2, and fit `y = theta(0) + theta(1)x + theta(2)x^2`, then we obtain a slightly better fit to the data. (Middle figure). Naively, it might seem that the more features we add, the better. However, there is also a danger in adding too many features: The rightmost figure is the result of fitting a 5th order polynomial. We see that even though the fitted curve passes through the data perfectly, we would not expect this to be a very good predictor of, say, housing prices (y) for different living areas (x). Without formally defining what these terms mean, we'll say the figure on the left shows an instance of **underfitting** - in which the data clearly shows structure not captured by the model - and the figure on the right is an example of **overfitting**.  

Underfitting, or high bias, is when the form of our hypothesis function h maps poorly to the trend of the data. It is usually caused by a function that is too simple or uses too few features. At the other extreme, overfitting, or high variance, is caused by a hypothesis function that fits the available data but does not generalize well to predict new data. It is usually caused by a complicated function that creates a lot of unnecessary curves and angles unrelated to the data.  

This terminology is applied to both linear and logistic regression. There are two main options to address the issue of overfitting:  
 * Reduce the number of features:
   * Manually select which features to keep
   * Use a model selection algorithm
 * Regularization
   * Keep all the features, but reduce the magnitude of parameters
   * Regularization works well when we have a lot of slightly useful features

## Cost Function
If we have overfitting from our hypothesis function, we can reduce the weight that some of the terms in our function carry by increasing their cost.  

Say we wanted to make the following function more quadratic:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/overfitting_1.png">  

We want to eliminate the influence of x^3 and x^4. Without actually getting rid of these features or changing the form of our hypothesis, we can instead modify our cost function:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/overfitting_2.png">  

We've added two extra terms at the end to inflate the cost of θ3​ and θ4​. Now, in order for the cost function to get close to zero, we will have to reduce the values of θ3​ and θ4​ to near zero. This will in turn greatly reduce the values of θ3x^3 and θ4x^4 in our hypothesis function. As a result, we see that the new hypothesis (depicted by the pink curve) looks like a quadratic function but fits the data better due to the extra small terms θ3x^3 and θ4x^4.  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/overfitting_3.png">  

We could aslo regularize all of our theta parameters in a single summation as:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/overfitting_4.png">  

The lamba, is the **regularization parameter**. It determines how much the costs of our theta parameters are inflated.  

## Regularized Linear Regression
We can apply regularization to both linear regression and logistic regression. We will aproach linear regression first.  

### Gradient Descent
We will modify our gradient descent function to separate out theta0 from the rest of the parameters because we do not want to penalize theta0.  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/overfitting_5.png">  

The term <img src="https://github.com/Pasoy/ml-projects/blob/master/images/overfitting_6.png"> performs our regularization. With some manipulation our update rule can also be represented as:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/overfitting_7.png">  

The first term in the above equation, `1−α * (λ/m)`​ will always be less than 1. Intuitively you can see it as reducing the value of `θj​` by some amount on every update. Notice that the second term is now exactly the same as it was before.  

### Normal Equation
To add in regularization, the equation is the same as our original, except that we add another term inside the parentheses:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/overfitting_8.png">  

L is a matrix with 0 at the top left and 1's down the diagonal, with 0's everywhere else. It should have dimension `(n+1)×(n+1)`. Intuitively, this is the identity matrix (though we are not including `x0`​), multiplied with a single real number `λ`.  

Recall that if m < n, then `transpose(X) * X` is non-invertible. However, when we add the term `λ * L`, then `transpose(X) * X + λ * L` becomes invertible.

## Regularized Logistic Regression
We can regularize logistic regression in a similar way that we regularize linear regression. As a result, we can avoid overfitting. The following image shows how the regularized function, displayed by the pink line, is less likely to overfit than the non-regularized function represented by the blue line:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/overfitting_9.png">  

### Cost Function
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/overfitting_10.png">  

#### MATLAB
```matlab
function [J, grad] = costFunction(theta, X, y)

m = length(y); % number of training examples

J = 0;
grad = zeroes(size(theta));


z = X * theta; % m x 1
hypothesis = sigmoid(z); % m x 1

J = -(1/m) * sum( (y .* log(hypothesis)) + ((1 - y) .* log(1 - hypothesis)) ); % scalar

grad = (1/m) * (X' * (hypothesis - y)); % (n+1) x 1

end
```
-----------

We can regularize this equation by adding a term to the end:   

```matlab
function [J, grad] = costFunctionReg(theta, X, y, lambda)

m = length(y); % number of training examples

J = 0;
grad = zeroes(size(theta));

z = X * theta; % m x 1
hypothesis = sigmoid(z); % m x 1

reg_term = (lambda / (2*m)) * sum(theta(2:end) .^ 2);

J = -(1/m) * sum( (y .* log(hypothesis)) + ((1 - y) .* log(1 - hypothesis)) ) + reg_term; % scalar

grad(1) = (1/m) * (X(:,1)' * (hypothesis - y); % 1 x 1
grad(2:end) = (1/m) * (X(:,2:end)' * (hypothesis - y)) + (lambda/m) * theta(2:end); % n x 1

end
```

The second sum **means to explicitly exclude** the bias term, theta0. i.e. the theta vector is indexed from 0 to n (holding n+1 values, theta0 through thetaN), and this sum skips theta0, by running from 1 to n, skipping 0. Thus, when computing the equation, we should continuosly update the following equations:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/overfitting_12.png">  

# Motivations

## Non-linear Hypothesis
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/nonlinear_hypothesis.png">  

As the number of features increase, the **number of terms in the hypothesis would also increase**, but there is a **probability of overfitting**. So for highly complex tasks like the ones where one needs to classify objects from images, logistic regression would not perform well.

e.g. for images of size 100 * 100 pixels if we use all quadratic features, there would be 50 million parameters to learn.
```matlab
m * n + m * n + C(m * n, 2) = 2m * n + C(m * n, 2)
```

This is where **Neural Networks** come in handy.

## Neurons and the Brain
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/neurons_ab.png">  

Origins:  
 * Algorithms that try to mimic the brain
 * Was very widely used in 80s and early 90s; popularity diminished in late 90s.
 * Recent regurence: State-of-the-art technique for many applications
 
### The "one learning algorithm" hypothesis
The brain has a **Auditory Cortex**. To hear, our ears pick up sound signals and route it to it. If we use our eyes and for example route the inputs to the **Auditory Cortex**, it learns to see.

# Neural Networks

## Model Representation 1
Neurons, are basically computational units that take inputs (**dendrites**) as electrical inputes (called "spikes" that are channeled to outputs (**axons**). In a simple model, the dendrites are like the input features `x1,..,xn` and the output is the result of `h(x)`. Our `x0` input node is sometimes called the "bias unit". It is always equal to 1. In neural networks, we use the same logistic function as in classification, yet we sometimes call it a sigmoid (logistic) **activation** function. Parameters are also called **weights**.  

A simple representation looks like:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/nn_1.png">  

Our input nodes, also known as the "input layer", go into another node, which finally outputs the hypothesis function, known as the "output layer".  
We can have intermediate layers of nodes between the input and output layers called the "hidden layers".  

In this example, we label these hidden layer nodes `a0^2 ... an^1` and call them "activation units".  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/nn_2.png">  

If we had one hidden layer:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/nn_3.png">  

The values for each of the "activation" nodes is obtained as follows:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/nn_4.png">  

This is saying that we compute our activation nodes by using a 3x4 matrix of parameters. We apply each row of the parameters to our inputs to obtain the value for one activation node. Our hypothesis output is the logistic function applied to the sum of the values of our activation nodes, which have been multiplied by yet another parameter matrix <img src="https://github.com/Pasoy/ml-projects/blob/master/images/nn_4_o2.png"> containing the weights for our second layer of nodes.  
Each layer gets its own matrix of weights, <img src="https://github.com/Pasoy/ml-projects/blob/master/images/nn_4_oj.png">.  
The dimensions of these matrices of weights is determined as follows:  

> If network has `sj` units in layer `j` and `sj+1` units in layer `j+1`, then <img src="https://github.com/Pasoy/ml-projects/blob/master/images/nn_4_oj.png"> will be of dimension <img src="https://github.com/Pasoy/ml-projects/blob/master/images/nn_4_dim.png">.  

The +1 comes from the addition in <img src="https://github.com/Pasoy/ml-projects/blob/master/images/nn_4_oj.png"> of the "bias nodes" `x0` and `theta0(j)`. In other words the output nodes will not include the bias nodes while the inputs will. The following image (by Andrew N.) summarizes the model representation:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/nn_5.png">  

### Example
If layer 1 has 2 input nodes and layer 2 has 4 activation nodes. Dimension of `theta(1)` is going to be 4x3 where `sj = 2` and `sj+1 = 4`.  

## Model Representation 2
Now we do a vectorized implementation of the functions. We're going to define a new variable <img src="https://github.com/Pasoy/ml-projects/blob/master/images/nn2_2.png"> that encompasses the parameters inside the function `g`. In the previous example if we replaced by the variable `z` for all the parameters we would get:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/nn2_3.png">  

In other words, for layer `j = 2` and node `k`, the variable `z` will be:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/nn2_4.png">  

The vector representation of `x` and <img src="https://github.com/Pasoy/ml-projects/blob/master/images/nn2_5.png"> is:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/nn2_6.png">  

Setting `x = a(1)`, we can rewrite the equation as:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/nn2_7.png">  

We are multiplying our matrix <img src="https://github.com/Pasoy/ml-projects/blob/master/images/nn2_8.png"> with the dimensions `sj * (n + 1)` (where `sj` is the number of our activation nodes) by our vector `a(j -1)` with the height `(n+1)`. This gives us our vector <img src="https://github.com/Pasoy/ml-projects/blob/master/images/nn2_10.png"> with height `sj`. Now we can get a vector of our activation nodes for layer `j` as follows:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/nn2_9.png">  

Where our function g can be applied element-wise to our vector <img src="https://github.com/Pasoy/ml-projects/blob/master/images/nn2_10.png">.  

We can then add a bias unit (equal to 1) to layer `j` after we have computed <img src="https://github.com/Pasoy/ml-projects/blob/master/images/nn2_11.png">. This will be element <img src="https://github.com/Pasoy/ml-projects/blob/master/images/nn2_12.png"> and will be equal to 1. To compute the final hypothesis, we have to first compute another vector `z`.  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/nn2_13.png">  

We get this vector by multiplying the next theta matrix after <img src="https://github.com/Pasoy/ml-projects/blob/master/images/nn2_16.png"> with the values of all the activation nodes we just got. This last theta matrix <img src="https://github.com/Pasoy/ml-projects/blob/master/images/nn2_14.png"> will have only **one row** which is multiplied by one column <img src="https://github.com/Pasoy/ml-projects/blob/master/images/nn2_11.png"> so that our result is a single number. We then get our final result with:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/nn2_15.png">

Notice that in this **last step**, between layer `j` and layer `j+1`, we are doing exactly the same thing as we did in logistic regression. Adding all these intermediate layers in neural networks allows us to more elegantly produce interesting and more complex non-linear hyptheses.  

## Examples and Intuitions

### One
The graph of our functions will look like:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/eai1_1.png">  

Set our first theta matrix as:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/eai1_2.png">  

This will cause the output of our hypothesis to only be positive if both `x1` and `x2` are 1. In other words:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/eai1_3.png">  

So we have constructed one of the fundamental operations in computers by using a small neural network rather than using an actual AND gate. Neural networks can also be used to simulate all other logical gates. The following is an example of the OR gate:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/eai1_4.png">  

### Two
The matrices for AND, NOR, and OR are:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/eai2_1.png">  

We can combine these to get the XNOR operator (which gives 1 if both inputs are 0 or 1)  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/eai2_2.png">  

For the transition between the first and second layer, we will use a matrix that combines the values for AND and NOR:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/eai2_3.png">  

For the transition between the second and third layer, we will use a matrix that uses the value for OR:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/eai2_4.png">  

Values for all the nodes:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/eai2_5.png">  

And so we have the XNOR operator using a hidden layer. Summary:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/eai2_6.png">

### Multiclass Classification
If we want to classify data into multiple classes, we let our hypothesis return a vector of values.  
We can define our set of resulting classes as y:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/eai3_1.png">  

Each `y(i)` represents a different image corresponding to the classes. The inner layers, each provide us with some new information which leads to our final hypothesis. The setup looks like:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/eai3_2.png">  

Our resulting hypothesis for one set of inputs may look like:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/eai3_3.png">  

In which case our resulting class if the third one down, or `h(x)3'`.  

# Cost Function and Backpropagation

## Cost Function
 * `L` = total number of layers in the network
 * `sl` = number of units (not counting bias unit) in layer l
 * `K` = number of output units/classes

In neural networks, we may have many outputs. We denote <img src="https://github.com/Pasoy/ml-projects/blob/master/images/cfab_1.png"> as being a hypothesis that results in the `k-th` output. The cost function for neural networks is going to be a generalization of logistic regression one. Recall that the cost function for regularized logistic regression was:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/cfab_2.png">  

For neural networks, it is going to be:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/cfab_3.png">  

Nested summations have been added to account for multiple output nodes. In the first part of the equation, before the square brackets, there is an additional nested summation that loops through the number of output nodes.  

In the regularization part, after the square brackets, it must account for multiple theta matrices. The number of columns in our current theta matrix is equal to the number of nodes in our current layer (including the bias unit). The number of rows in our current theta matrix is equal to the number of nodes in the next layer (excluding the bias unit). As before with logistic regression, every term is squared.  

Note:  
 * the double sum simply adds up the logistic regression costs calculated for each cell in the output layer
 * the triple sum simply adds up the squares of all the individual thetas in the entire network
 * the **i** in the triple sum does not refer to training example **i**

## Backpropagation Algorithm
Backpropagation is neural-network terminology for minimizing the cost function. Goal is to compute:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/cfab_4.png">  

That is, we want to minimize our cost function `J` using an optimal set of parameters in theta. Now we will look at the equations to use to compute the partial derivative of `J(theta)`:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/cfab_5.png">  

To do so, use the following algorithm:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/cfab_6.png">  

Given training set <img src="https://github.com/Pasoy/ml-projects/blob/master/images/cfab_7.png">  
 * Set <img src="https://github.com/Pasoy/ml-projects/blob/master/images/cfab_8.png"> for all `(l, i, j)`, (hence you end up having a matrix full of zeros)

For training example `t = 1` to m:  
 1. Set `a(1) := x(t)`
 2. Perform forward propagation to compute `a(l)` for `l = 2, 3, .., L`

<img src="https://github.com/Pasoy/ml-projects/blob/master/images/cfab_9.png">  

 3. Using `y(t)`, compute <img src="https://github.com/Pasoy/ml-projects/blob/master/images/cfab_10.png">  

Where `L` is our total number of layers and `a(L)` is the vector of outputs of the activation units for the last layer. So our "error values" for the last layer are simply the differences of our actual results in the last layer and the correct outputs in `y`. To get the delta values of the layers before the last layer, we can use an equation that steps back from right to left:  

 4. Compute <img src="https://github.com/Pasoy/ml-projects/blob/master/images/cfab_11.png"> using <img src="https://github.com/Pasoy/ml-projects/blob/master/images/cfab_12.png">

The delta values of layer `l` are calculated by multiplying the delta values in the next layer with the theta matrix of layer `l`. Then element-wise multiply that with a function called `g'`, or **g-prime**, which is the derivative of the activation function `g` evaluated with the input values given by `z(l)`

The **g-prime** derivative terms can also be written out as:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/cfab_13.png">  

 5. <img src="https://github.com/Pasoy/ml-projects/blob/master/images/cfab_14.png"> or with vectorization, <img src="https://github.com/Pasoy/ml-projects/blob/master/images/cfab_15.png">  
 
Hence we update our new capital-delta matrix
 * <img src="https://github.com/Pasoy/ml-projects/blob/master/images/cfab_16.png">  
 * <img src="https://github.com/Pasoy/ml-projects/blob/master/images/cfab_17.png">  

The capital-delta matrix D is used as an "accumulator" to add up our values as we go along and eventually compute our partial derivative. Thus we get:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/cfab_18.png">  

## Backpropagation Intuition
If we consider simple non-multiclass classification (k = 1) and disregard regularization, the cost is computed with:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/cfab_19.png">  

Intuitively, <img src="https://github.com/Pasoy/ml-projects/blob/master/images/cfab_20.png">​ is the "error" for <img src="https://github.com/Pasoy/ml-projects/blob/master/images/cfab_21.png"> (unit j in layer l). More formally, the delta values are actually the derivative of the cost function:  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/cfab_22.png">  

Recall that our derivative is the slope of a line tangent to the cost function, so the steeper the slope the more incorrect we are. Let us consider the following neural network below and see how we could calculate some <img src="https://github.com/Pasoy/ml-projects/blob/master/images/cfab_20.png">  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/cfab_23.png">  