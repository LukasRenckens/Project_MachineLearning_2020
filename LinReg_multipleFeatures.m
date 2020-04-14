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
cv = data_array([30817:size(data_array)], :);      % Cross validation set 20%

%training = training(1:1000,:);                      % Take first few

y = training(:,1);
X = training(:, [2 3]);                             % Odometer_value | year_produced

m = length(y);                                           % Number of training examples

[X mu sigma] = featureNormalize(X);                      % Normalize every feature ~ -3<X<+3    
X = [ones(m, 1) X];                                      % Add a column of ones (x0) to X

%% Plot data



%% Cost and Gradient descent

% Choose some alpha value
alpha = 0.1;
num_iters = 100;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);

[theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 1);
xlabel('Number of iterations');
ylabel('Cost J');

fprintf('== Gradient decent ==\n');

% Display gradient descent's result
fprintf('Theta:\n');
fprintf(' %f \n', theta);

J = computeCost(X, y, theta);
fprintf('Cost = %f\n', J);

% Estimate the price of a car with: 
x0 = 1;
x1 = (1000 - mu(1,1))/sigma(1,1);
x2 = (2000 - mu(1,2))/sigma(1,2);

price = theta(1,1)*x0 + theta(2,1)*x1 + theta(3,1)*x2; 

fprintf(['Predicted price: $%f\n'], price);
fprintf('\n');
%% Normal equation

y = training(:, 1);
X = training(:, 2:3);  
X = [ones(m, 1) X];   

%No feature scaling needed for normal equation!

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

fprintf('== Normal equation ==\n');

% Display normal equation's result
fprintf('Theta:\n');
fprintf(' %f \n', theta);

J = computeCost(X, y, theta);
fprintf('Cost = %f\n', J);

% Estimate the price of a car with: 
x0 = 1;
x1 = 1000;
x2 = 2000;

price = theta(1,1)*x0 + theta(2,1)*x1 + theta(3,1)*x2; 

fprintf(['Predicted price: $%f\n'], price);
