function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

J = (sum((X * theta - y) .^ 2) + lambda * sum(theta(2:end) .^ 2)) / (2 * m);
grad = ((theta' * X' - y') * X)' / m + lambda / m * [0; theta(2:end)];
end