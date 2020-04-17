%% Initialize
clear ; close all;

%% Read data
data_table = readtable('cars_custom.txt');

% price_usd | Odometer_value | year_produced | engine_capacity              %| nr_of_photos | up_counter | duration_listed
data_array = table2array(data_table(:, [15 5 6 10]));                       % 17 18 29

% Add numbering column
% numbers_array = [1:size(data_array)]';
% data_array = cat(2, numbers_array, data_array); 

%% Process data
data_array = data_array(randperm(size(data_array,1)),:); % Randomize order

training = data_array([1:23113], :);               % Trainging set 60%
test = data_array([23114:30817], :);               % Test set 20%
cv = data_array([30818:size(data_array)], :);      % Cross validation set 20%

% Take first few data
m = 100;
training = training(1:m,:); 
test = test(1:m,:);
cv = cv(1:m,:);

y = training(:,1);
X = training(:,3);                       % Year produced  
ytest = test(:,1);
Xtest = test(:,3);
ycv = cv(:,1);
Xcv = cv(:,3);

m = length(y);                           % number of training examples

%% Ploynomial features
p = 8;

% Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p);
[X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
X_poly = [ones(m, 1), X_poly];                   % Add Ones

% Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p);
X_poly_test = bsxfun(@minus, X_poly_test, mu);
X_poly_test = bsxfun(@rdivide, X_poly_test, sigma);
X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];         % Add Ones


% Map X_poly_val and normalize (using mu and sigma)
X_poly_cv = polyFeatures(Xcv, p);
X_poly_cv = bsxfun(@minus, X_poly_cv, mu);
X_poly_cv = bsxfun(@rdivide, X_poly_cv, sigma);
X_poly_cv = [ones(size(X_poly_cv, 1), 1), X_poly_cv];           % Add Ones


% Add a column of ones to x
X = [ones(m, 1), X];     
Xtest = [ones(size(Xtest, 1), 1), Xtest];
Xcv = [ones(size(Xcv, 1), 1), Xcv];

%% Plot data
% plotData(training(:,2), training(:,1), 'Odometer value');
% plotData(training(:,3), training(:,1), 'Production year');
% plotData(training(:,4), training(:,1), 'Engine capacity');
% plotData(training(:,5), training(:,1), 'Number of photos');
% plotData(training(:,6), training(:,1), 'Up counter');
% plotData(training(:,7), training(:,1), 'Duration listed');

plotData(X(:,2), y, 'Production year');

%% Cost and Gradient descent
% Some gradient descent settings
% iterations = 1500;
% alpha = 0.1;
lambda = 0.003;

% initialize theta
% theta = zeros(2, 1);    

fprintf('\nRunning Gradient Descent ...\n')
% run gradient descent
%theta = gradientDescent(X, y, theta, alpha, iterations, lambda);
theta = trainLinearReg(X_poly, y, lambda);

% print theta to screen
fprintf('Theta found by gradient descent:\n');
fprintf('%f\n', theta);

J = computeCost(X_poly, y, theta, lambda);
fprintf('Cost computed = %f\n', J);

% Plot the linear fit
hold on; % keep previous plot visible
x = (min(X(:,2)): 0.05 : max(X(:,2)))';

% Map the X values 
X_poly_plot = polyFeatures(x, p);
X_poly_plot = bsxfun(@minus, X_poly_plot, mu);
X_poly_plot = bsxfun(@rdivide, X_poly_plot, sigma);

% Add ones
X_poly_plot = [ones(size(x, 1), 1) X_poly_plot];

% Plot
plot(x, X_poly_plot * theta, '-b')
legend('Training data', 'Polynomial regression')
hold off % don't overlay any more plots on this figure

%Plot the learning curve
[error_train, error_cv] = learningCurve(X_poly, y, X_poly_cv, ycv, lambda);
