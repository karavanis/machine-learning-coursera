function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);


% You need to return the following values correctly
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

M = (X * Theta' - Y) .^ 2;
J = sum(sum(R .* M)) / 2 + lambda / 2 * sum(sum(Theta .^ 2)) + lambda / 2 * sum(sum(X .^ 2));

for i = 1:num_movies
  X_grad(i, :) = (R(i, :) .* (X(i, :) * Theta' - Y(i, :)) * Theta) + lambda * X(i, :);
end

for j = 1:num_users
  Theta_grad(j, :) = (R(:, j)' .* (Theta(j, :) * X' - Y(:, j)') * X) + lambda * Theta(j, :);
end

grad = [X_grad(:); Theta_grad(:)];
end
