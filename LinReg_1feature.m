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

%training = training(1:100,:);                % Take first few

% Split up and name data_array columns
price_array = training(:,1);
odo_array = training(:,2);
year_array = training(:,3);
engine_array = training(:,4);
photos_array = training(:,5);
counter_array = training(:,6);
duration_array = training(:,7);

y = price_array;
X = year_array;
m = length(y);                                           % number of training examples
[X mu sigma] = featureNormalize(X);                      % Normalize every feature ~ -3<X<+3    
X = [ones(m, 1), X];                                     % Add a column of ones to x
theta = zeros(2, 1);                                     % initialize fitting parameters

%% Plot data
plotData(odo_array, price_array, 'Odometer value');
plotData(year_array, price_array, 'Production year');
plotData(engine_array, price_array, 'Engine capacity');
plotData(photos_array, price_array, 'Number of photos');
plotData(counter_array, price_array, 'Up counter');
plotData(duration_array, price_array, 'Duration listed');

plotData(X(:,2), y, 'Production year');

%% Cost and Gradient descent

% Some gradient descent settings
iterations = 1500;
alpha = 0.1;

fprintf('\nRunning Gradient Descent ...\n')
% run gradient descent
theta = gradientDescent(X, y, theta, alpha, iterations);

% print theta to screen
fprintf('Theta found by gradient descent:\n');
fprintf('%f\n', theta);

J = computeCost(X, y, theta);
fprintf('Cost computed = %f\n', J);

% Plot the linear fit
hold on; % keep previous plot visible
plot(X(:,2), X*theta, '-b')
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure

%% Odometer value

y = price_array;
X = odo_array;
m = length(y);                                           % number of training examples
[X mu sigma] = featureNormalize(X);                      % Normalize every feature ~ -3<X<+3    
X = [ones(m, 1), X];                                     % Add a column of ones to x
theta = zeros(2, 1);                                     % initialize fitting parameters

%% Plot data
% plotData(odo_array, price_array, 'Odometer value');
% plotData(year_array, price_array, 'Production year');
% plotData(engine_array, price_array, 'Engine capacity');
% plotData(photos_array, price_array, 'Number of photos');
% plotData(counter_array, price_array, 'Up counter');
% plotData(duration_array, price_array, 'Duration listed');

plotData(X(:,2), y, 'Odometer value');

%% Cost and Gradient descent

% Some gradient descent settings
iterations = 1500;
alpha = 0.1;

fprintf('\nRunning Gradient Descent ...\n')
% run gradient descent
theta = gradientDescent(X, y, theta, alpha, iterations);

% print theta to screen
fprintf('Theta found by gradient descent:\n');
fprintf('%f\n', theta);

J = computeCost(X, y, theta);
fprintf('Cost computed = %f\n', J);

% Plot the linear fit
hold on; % keep previous plot visible
plot(X(:,2), X*theta, '-b')
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure

