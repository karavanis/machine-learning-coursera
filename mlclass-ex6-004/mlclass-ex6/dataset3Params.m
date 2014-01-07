function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

values = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
num_of_values = numel(values);

% set the value to a really high error that is impossible to reach
smallest_error = 10^10000;

for i = 1:num_of_values
  for j = 1:num_of_values
    model = svmTrain(X, y, values(i), @(x1, x2) gaussianKernel(x1, x2, values(j)));
    predictions = svmPredict(model, Xval);
    error = mean(double(predictions ~= yval));

    if error < smallest_error
      smallest_error = error;
      C = values(i);
      sigma = values(j);
    end
  end
end
end
