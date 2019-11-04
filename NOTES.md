# Machine Learning
Here are some notes I took from doing courses/reading books.

## What is Machine Learning?
> "A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E." - Tom Mitchell

e.g. playing checkers  
E = experience of playing many games  
T = task of playing checkers  
P = probability that the program will win the next game

## Supervised learning
We are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and the output.

 * right answers given
 * regression (goal is to predict a continuous valued output)
 * classification (output 0 or 1)

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

 * alpha is the learning rate (controls how big a step is)
 * alpha is always a positive number
 * term after alpha is called the derivative
 * := is an assignment
 * j represents the feature index number  

In gradient descent we simultaneously update θ0 and θ1.  
If already at the local minimum it will not change.

### Learning rate (alpha)
 * is it is too small, gradient descent can be slow
 * if it is too large, gradient descent can overshoot the minimum. it may fail to converge or even diverge
 * after descent, it will automatically take smaller steps
 
### "Batch" gradient descent
Means each step of gradient descent uses all training data.

***

# Linear algebra

## Matrix
 * is an rectangular array of numbers.  
 * dimension of matrix: number of rows * number of columns.  
 * usually uppercase
 
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
to add matrices, they have to be the same dimension.  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/matrix_addition.png"> 

### Scalar multiplication/division
*scalar is a real number*  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/matrix_scalar.png">  

### Multiplication with a vector
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/matrix_vector_multi.png">  

 * result is a vector
 * number of columns of the matrix must equal the number of rows of the vector

Here is a website for good visualization: http://matrixmultiplication.xyz/

### Matrix matrix multiplication
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/matrix_matrix_multi.png">  

 * result is a matrix (dimension: M1 columns * M2 rows)
 * number of columns in the first matrix must match the number of rows in the second matrix
 * can be used to predict something (if `h(x)` is given and sample data; see below)
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/matrix_matrix_multi_predictions.png">

### Identity matrix
 * denoted `I (or In*n)`
 * it can be a different dimension
 * for any matrix A, `A*I = I*A = A`
 
### Inverse
 * if a matrix is a square matrix (m*m), and has an inverse, then `A*(A^(-1)) = I`. I being the identity matrix
 * matrices which don't have an inverse are "**singular**" or "**degenerate**"
 
### Transpose
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/matrix_transpose_1.png">  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/matrix_transpose_2.png">  

 * means sort of flipping the matrix
 * denoted `A *superscript* T`
 * rows become the columns
 * columns become rows
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
 * usually lowercase
 * is an `n * 1` matrix
 * dimension = number of rows
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
 
### Formula
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/reg_new_hypothesis.png">  
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/reg_new_hypothesis_1.png">  

## Gradient descent for multiple variables
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/gradient_descent_new.png">  

### Feature scaling
the idea is that the features are on a similar scale.   
 * it takes less time, because it will descend quickly on small ranges
 * more accurate
 * less complicated  

get every feature into a `-1 =< x(i) <= 1` range. of course, it can be a bit bigger.

### Mean normalization
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/gradient_descent_mean.png">  

replace `x(i) with x(i) - μ(i)`. μ being the average value of the training examples. also divide by `s(i)` (the range of values (max-min) ).  
```
e.g we have an algorithm to estimate the life span of a tree.
in our training set, we have trees with the life span between 5 and 30 years. our average is 13 years.
x(i) = (life span - 13) / 25
```
