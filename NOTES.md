# Machine Learning
Here are some notes I took from doing courses/reading books.

## What is Machine Learning?
"A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E." - Tom Mitchell

e.g. playing checkers

E = experience of playing many games

T = task of playing checkers

P = probability that the program will win the next game

## Supervised learning
We are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and the output.

 * right answers given
 * regression (goal is to predict a continuous valued output)
 * classification (output 0 or 1)

## Unsupervised learning
We are given data, but don't know what to do with it. Let the algorithm figure it out.

### clustering algorithm

 * Given a set of news articles, automatically group them into sets of articles about the same story
 * Given a database of customer data, discover market segments and group customers into different market segments

### cocktail party algorithm
find structure in a chatoic environment

```
[W,s,v] = svd((repmat(sum(x.*x, 1), size(x, 1), 1).*x)*x^i);
```
 * svd = single value decomposition
 * repmat = replicate and tile an array
