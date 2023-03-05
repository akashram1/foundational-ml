# foundational-ml
Implementations of various basic ML algorithms.

## List
### Linear Models
All could be made non-linear using non-linear features (like `X1^2`, `log(X1)`). The model will be non-linear in terms of 
X1 but linear in terms of `X3` and `X4` where `X3 = X1^2` and `X4 = log(X1)`
1. Linear Regression
2. Logistic Regression
3. Perceptron


### Neat sklearn methods
Helpful in debugging models with a small toy dataset
| method        | purpose       | 
| ------------- |:-------------:|
| [`make_blobs`] (https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html)   | Create points based on a Gaussian distribution. Great to create linearly separable toy datasets|