# Machine Learning
Here are some notes I took from doing courses/reading books.

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