function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

num_of_elem = size(X, 1);

% You need to return the following variables correctly.
idx = zeros(num_of_elem, 1);

for i = 1:num_of_elem
  min_distance = 10^10;
  for j = 1:K
    current_distance = norm(X(i, :) - centroids(j, :));
    if min_distance > current_distance
      idx(i) = j;
      min_distance = current_distance;
    end
  end
end
end

