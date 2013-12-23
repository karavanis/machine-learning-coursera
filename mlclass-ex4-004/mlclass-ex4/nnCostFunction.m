function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

X = [ones(m, 1) X];

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


Y = zeros(m, num_labels);
for i = 1: m
  Y(i, y(i)) = 1;
end

a1 = X;
z2 = a1 * Theta1';
a2 = sigmoid(z2);

a2 = [ones(size(a2, 1), 1) a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);

J = sum(sum(-Y .* log(a3) - (1 - Y) .* log(1 - a3))) / m;

regularization_term = lambda * (sum(sum(Theta1(:, 2: end) .^ 2)) + sum(sum(Theta2(:, 2: end) .^ 2))) / (2 * m);

J = J + regularization_term;

d3 = a3 - Y;
d2 = d3 * Theta2(:, 2: end) .* sigmoidGradient(z2);

Theta2_grad = (d3' * a2 + lambda .* [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)]) / m;
Theta1_grad = (d2' * a1 + lambda .* [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)]) / m;

% size(Theta2)
% size(Theta2_grad)
% size(Theta2,1)
% size(Theta2,2)
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
