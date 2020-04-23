%% Initialize
clear ; close all;

%% Read data
data_table = readtable('cars_custom_rand.txt');

% price_usd | Odometer_value | year_produced | engine_capacity | nr_of_photos | up_counter | duration_listed
data_array = table2array(data_table(:, [23:29])); 

% Add numbering column
% numbers_array = [1:size(data_array)]';
% data_array = cat(2, numbers_array, data_array); 

%% Process data
training = data_array([1:23113], :);               % Trainging set 60%
test = data_array([23114:30817], :);               % Test set 20%
cv = data_array([30818:size(data_array)], :);      % Cross validation set 20%

% Take first few data
% m = 100;
% training = training(1:m,:); 
% test = test(1:m,:);
% cv = cv(1:m,:);

y = training(:,1);
X = training(:,3);                       % Year produced  
ytest = test(:,1);
Xtest = test(:,3);
ycv = cv(:,1);
Xcv = cv(:,3);

m = length(y);                           % number of training examples 

% Normalize every feature ~ -3<X<+3  
[X_reg mu sigma] = featureNormalize(X);

% Add a column of ones to x
X = [ones(m, 1), X]; 
X_reg = [ones(m, 1), X_reg];
Xtest = [ones(size(Xtest, 1), 1), Xtest];
Xcv = [ones(size(Xcv, 1), 1), Xcv];

%% Plot data
plotData(training(:,2), training(:,1), 'Odometer value');
% plotData(training(:,3), training(:,1), 'Production year');
plotData(training(:,4), training(:,1), 'Engine capacity');
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
theta = zeros(3, 1);    

fprintf('\nRunning Gradient Descent ...\n')
% run gradient descent
%theta = gradientDescent(X, y, theta, alpha, iterations, lambda);
theta = trainLinearReg(X_reg, y, lambda);

% print theta to screen
fprintf('Theta found by gradient descent:\n');
fprintf('%f\n', theta);

J = computeCost(X_reg, y, theta, lambda);
fprintf('Cost computed = %f\n', J);

%% Plot results
%Plot the linear fit
hold on; % keep previous plot visible
plot(X(:,2), X_reg*theta, '-b')
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure

