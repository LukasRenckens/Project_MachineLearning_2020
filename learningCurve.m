function [error_train, error_cv] = learningCurve(X, y, Xcv, ycv, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);

error_train = zeros(m, 1);
error_cv   = zeros(m, 1);

for i = 1:m
  Xtrain = X(1:i, :);
  ytrain = y(1:i);
  theta = trainLinearReg(Xtrain, ytrain, lambda);
  error_train(i)= computeCost(Xtrain, ytrain, theta, 0);  
  error_cv(i)= computeCost(Xcv, ycv, theta, 0);
end

%Plot curves
% 2 seperate curves
% figure;
% plot(1:size(X, 1), error_train);
% title('Training')
% xlabel('Number of training examples')
% ylabel('Error')
% figure;
% plot(1:size(X, 1), error_cv);
% title('Cross validation');
% xlabel('Number of training examples')
% ylabel('Error')

% 1 curve
figure;
plot(1:size(X, 1), error_train, 1:size(X, 1), error_cv);
title('Learning curve for linear regression')
xlabel('Number of training examples')
ylabel('Error')
legend('Train', 'Cross Validation')

end
