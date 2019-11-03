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
Calculation is the hypothesis value for h(x), minus the actual value of y and then we square.  
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

### Formula 
<img src="https://github.com/Pasoy/ml-projects/blob/master/images/gradient_descent.png">  
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
