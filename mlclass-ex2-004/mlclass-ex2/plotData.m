function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

for i = 1:length(y)
  if (y(i) == 0)
    symbol = 'ko';
  else
    symbol = 'k+';
  endif
  plot(X(i, 1), X(i, 2), symbol);
end

hold off;

end
