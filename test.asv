%% Initialize
clear ; close all;

%% Read data
data_table = readtable('cars.txt');

% Odometer_value | year_produced | engine_capacity | price_usd | nr_of_photos | up_counter | duration_listed
data_array = table2array(data_table(:, [5 6 10 15 17 18 29]));  

% Add numbering column
numbers_array = [1:size(data_array)]';
data_array = cat(2, numbers_array, data_array); 

%% Process data
%data_array = data_array(randperm(size(data_array,1)),:); % Randomize order
data_array = data_array(1:100,:);                        % Take first few

% Split up and name data_array columns
odo_array = data_array(:,2);
year_array = data_array(:,3);
engine_array = data_array(:,4);
price_array = data_array(:,5);
photos_array = data_array(:,6);
counter_array = data_array(:,7);
duration_array = data_array(:,8);

y = price_array/10000;
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

%% Plot data odo value

plotData(X2(:,2), y, 'Odometer value');
%% Cost and Gradient descent odo value

% Some gradient descent settings
iterations = 1500;
alpha = 0.0036;

fprintf('\nRunning Gradient Descent ...\n')
% run gradient descent
theta = gradientDescent(X2, y, theta, alpha, iterations);

% print theta to screen
fprintf('Theta found by gradient descent:\n');
fprintf('%f\n', theta);

J = computeCost(X2, y, theta);
fprintf('Cost computed = %f\n', J);

% Plot the linear fit
hold on; % keep previous plot visible
plot(X2(:,2), X1*theta, '-b')
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure
