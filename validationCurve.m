function [lambda_vec, error_train, error_cv] = validationCurve(X, y, Xcv, ycv)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%

% Selected values of lambda (you should not change this)
lambda_vec = [0 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10 30 100]';

error_train = zeros(length(lambda_vec), 1);
error_cv = zeros(length(lambda_vec), 1);

Xtrain = X;
ytrain = y;

for i = 1:length(lambda_vec)
  lambda = lambda_vec(i);
  theta = trainLinearReg(Xtrain, ytrain, lambda);
  error_train(i)= computeCost(Xtrain, ytrain, theta, 0);  
  error_cv(i)= computeCost(Xcv, ycv, theta, 0);
end

% Plot curve
figure;
plot(lambda_vec, error_train, lambda_vec, error_cv);
%semilogx(lambda_vec, error_train, lambda_vec, error_cv);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');
title('Validation curve for selecting lambda');

end
