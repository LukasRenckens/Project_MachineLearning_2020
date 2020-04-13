%% Initialize
clear ; close all;

%% Read data
data_table = readtable('cars_custom.txt');

% Odometer_value | year_produced | engine_capacity | price_usd | nr_of_photos | up_counter | duration_listed
data_array = table2array(data_table(:, [5 6 10 15 17 18 29]));  

% Add numbering column
numbers_array = [1:size(data_array)]';
data_array = cat(2, numbers_array, data_array); 

%% Process data
data_array = data_array(randperm(size(data_array,1)),:); % Randomize order

training_array = data_array([1:23113], :);               % Trainging set 60%
test_array = data_array([23114:30817], :);               % Test set 20%
cv_array = data_array([30817:size(data_array)], :);      % Cross validation set 20%

training_array = training_array(1:100,:);                % Take first few

% Split up and name data_array columns
odo_array = training_array(:,2);
year_array = training_array(:,3);
engine_array = training_array(:,4);
price_array = training_array(:,5);
photos_array = training_array(:,6);
counter_array = training_array(:,7);
duration_array = training_array(:,8);

y = price_array;
m = length(y); % number of training examples
X1 = [ones(m, 1), (year_array(:,1)/1000)]; % Add a column of ones to x
X2 = [ones(m, 1), (odo_array(:,1)/10000)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters

%% Plot data
% plotData(odo_array, price_array, 'Odometer value');
% plotData(year_array, price_array, 'Production year');
% plotData(engine_array, price_array, 'Engine capacity');
% plotData(photos_array, price_array, 'Number of photos');
% plotData(counter_array, price_array, 'Up counter');
% plotData(duration_array, price_array, 'Duration listed');

plotData(X1(:,2), y, 'Production year');

%% Cost and Gradient descent

% Some gradient descent settings
iterations = 1500;
alpha = 1.246;

fprintf('\nRunning Gradient Descent ...\n')
% run gradient descent
theta = gradientDescent(X1, y, theta, alpha, iterations);

% print theta to screen
fprintf('Theta found by gradient descent:\n');
fprintf('%f\n', theta);

J = computeCost(X1, y, theta);
fprintf('Cost computed = %f\n', J);

% Plot the linear fit
hold on; % keep previous plot visible
plot(X1(:,2), X1*theta, '-b')
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure
