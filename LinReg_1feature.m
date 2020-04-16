%% Initialize
clear ; close all;

%% Read data
data_table = readtable('cars_custom.txt');

% price_usd | Odometer_value | year_produced | engine_capacity | nr_of_photos | up_counter | duration_listed
data_array = table2array(data_table(:, [15 5 6 10 17 18 29]));   

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

[X mu sigma] = featureNormalize(X);      % Normalize every feature ~ -3<X<+3  

% Add a column of ones to x
X = [ones(m, 1), X];     
Xtest = [ones(size(Xtest, 1), 1), Xtest];
Xcv = [ones(size(Xcv, 1), 1), Xcv];

%% Ploynomial features
% p = 2;
% 
% % Map X onto Polynomial Features and Normalize
% X = polyFeatures(X, p);
% [X, mu, sigma] = featureNormalize(X);  % Normalize
% X = [ones(m, 1), X];                   % Add Ones
% 
% % Map X_poly_test and normalize (using mu and sigma)
% Xtest = polyFeatures(Xtest, p);
% Xtest = bsxfun(@minus, Xtest, mu);
% Xtest = bsxfun(@rdivide, Xtest, sigma);
% Xtest = [ones(size(Xtest, 1), 1), Xtest];         % Add Ones
% 
% 
% % Map X_poly_val and normalize (using mu and sigma)
% Xcv = polyFeatures(Xcv, p);
% Xcv = bsxfun(@minus, Xcv, mu);
% Xcv = bsxfun(@rdivide, Xcv, sigma);
% Xcv = [ones(size(Xcv, 1), 1), Xcv];           % Add Ones
% 

% Add a column of ones to x
% X = [ones(m, 1), X];     
% Xtest = [ones(size(Xtest, 1), 1), Xtest];
% Xcv = [ones(size(Xcv, 1), 1), Xcv];

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
iterations = 1500;
alpha = 0.1;
lambda = 0;

% initialize theta
theta = zeros(2, 1);    

fprintf('\nRunning Gradient Descent ...\n')
% run gradient descent
%theta = gradientDescent(X, y, theta, alpha, iterations, lambda);
theta = trainLinearReg(X, y, lambda);

% print theta to screen
fprintf('Theta found by gradient descent:\n');
fprintf('%f\n', theta);

J = computeCost(X, y, theta, lambda);
fprintf('Cost computed = %f\n', J);

% Plot the linear fit
hold on; % keep previous plot visible
plot(X(:,2), X*theta, '-b')
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure

%Plot the learning curve
[error_train, error_cv] = learningCurve(X, y, Xcv, ycv, lambda);

%% Odometer value
% 
% y = price_array;
% X = odo_array;
% m = length(y);                                           % number of training examples
% [X mu sigma] = featureNormalize(X);                      % Normalize every feature ~ -3<X<+3    
% X = [ones(m, 1), X];                                     % Add a column of ones to x
% theta = zeros(2, 1);                                     % initialize fitting parameters
%
% %% Plot data
% 
% plotData(X(:,2), y, 'Odometer value');
% 
% %% Cost and Gradient descent
% 
% % Some gradient descent settings
% iterations = 1500;
% alpha = 0.1;
% 
% fprintf('\nRunning Gradient Descent ...\n')
% % run gradient descent
% %theta = gradientDescent(X, y, theta, alpha, iterations, lambda);
% theta = trainLinearReg(X, y, lambda);
% 
% % print theta to screen
% fprintf('Theta found by gradient descent:\n');
% fprintf('%f\n', theta);
% 
% J = computeCost(X, y, theta, lambda);
% fprintf('Cost computed = %f\n', J);
% 
% % Plot the linear fit
% hold on; % keep previous plot visible
% plot(X(:,2), X*theta, '-b')
% legend('Training data', 'Linear regression')
% hold off % don't overlay any more plots on this figure
% 
% 


