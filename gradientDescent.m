function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters, lambda)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    gradJ = 1/(2*m) * 2 * (X'*X*theta - X'*y);
    theta = theta - alpha * gradJ;

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta, lambda);

end

end
