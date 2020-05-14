%% Initialize
clear ; close all;

%% Read data
data_table = readtable('cars_custom_rand.txt');                            % Read randomized data

% Filter car_manufacturer, model_name, transmission, engine_fuel, drivetrain, duration_listed
% data_table = data_table(data_table.Var1=="Volkswagen" & data_table.Var29<100, :); 
data_table = data_table(data_table.Var1=="Volkswagen" & data_table.Var2=="Passat" & data_table.Var29<100, :); 
% data_table = data_table(data_table.Var1=="Volkswagen" & data_table.Var2=="Passat" & data_table.Var3=="mechanical" & data_table.Var5=="gasoline" & data_table.Var11=="front" & data_table.Var29<100, :); 

% price_usd | Odometer_value | year_produced | engine_capacity             %| nr_of_photos | up_counter | duration_listed
data_array = table2array(data_table(:, [23 24 25 26])); 

%% Process data
c1 = floor(size(data_array,1)/100*60);
c2 = c1+1;
c3 = c2 + floor(size(data_array,1)/100*20);
c4 = c3 + 1;
training = data_array([1:c1], :);                % Trainging set 60%
test = data_array([c2:c3], :);                   % Test set 20%
cv = data_array([c4:size(data_array,1)], :);     % Cross validation set 20%

% Take first few data
% m = 100;
% training = training(1:100,:);   
% test = test(1:100,:);
% cv = cv(1:100,:);

% Odometer_value | year_produced | engine_capacity 
y = training(:,1);
X = training(:,[2 3 4]);  
ytest = test(:,1);
Xtest = test(:,[2 3 4]); 
ycv = cv(:,1);
Xcv = cv(:,[2 3 4]); 

m = length(y);                                     % Number of training examples

[X mu sigma] = featureNormalize(X);                % Normalize every feature ~ -3<X<+3

% Add a column of ones (x0) to X
X = [ones(m, 1) X];   
Xtest = [ones(size(Xtest, 1), 1), Xtest];
Xcv = [ones(size(Xcv, 1), 1), Xcv];

%% Cost and Gradient descent
% Choose some alpha value
alpha = 0.3;
num_iters = 100;
lambda = 0.003;

% Init Theta and Run Gradient Descent 
theta = zeros(4, 1);

%[theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters, lambda);
theta = trainLinearReg(X, y, lambda);

% Plot the convergence graph
% figure;
% plot(1:numel(J_history), J_history, '-b', 'LineWidth', 1);
% title (sprintf('Curve for selecting alpha (alpha = %f)', alpha));
% xlabel('Number of iterations');
% ylabel('Cost J');

fprintf('== Gradient decent ==\n');

% Display gradient descent's result
fprintf('Theta:\n');
fprintf(' %f \n', theta);

% Compute cost for test set
J = computeCost(Xtest, ytest, theta, lambda);
fprintf('Cost = %f\n', J);

%Plot the learning curve
%[error_train, error_cv] = learningCurve(X, y, Xcv, ycv, lambda);

% Estimate the price of a car with: 
% Normalized!
x0 = 1;                                 % x0 = 1
x1 = (100000 - mu(1,1))/sigma(1,1);      % x1 = Odometer value 
x2 = (2000 - mu(1,2))/sigma(1,2);       % x2 = Production year 
x3 = (2 - mu(1,3))/sigma(1,3);          % x3 = engine capacity

price = theta(1,1)*x0 + theta(2,1)*x1 + theta(3,1)*x2 + theta(4,1)*x3; 

fprintf(['Predicted price: $%f\n'], price);
fprintf('\n');

%% Normal equation
y = training(:, 1);
X = training(:, [2 3 4]);  
X = [ones(m, 1) X];   

%No feature scaling needed for normal equation!

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

fprintf('== Normal equation ==\n');

% Display normal equation's result
fprintf('Theta:\n');
fprintf(' %f \n', theta);

% Compute cost for test set
J = computeCost(Xtest, ytest, theta, lambda);
fprintf('Cost = %f\n', J);

% Estimate the price of a car with: 
x0 = 1;                 % x0 = 1
x1 = 100000;            % x1 = Odometer value 
x2 = 2000;              % x2 = Production year 
x3 = 2;                 % x3 = engine capacity

price = theta(1,1)*x0 + theta(2,1)*x1 + theta(3,1)*x2 + theta(4,1)*x3; 

fprintf(['Predicted price: $%f\n'], price);

%Plot the learning curve
%[error_train, error_cv] = learningCurve(X, y, Xcv, ycv, lambda);

% Plot validation curve for selecting lambda
%[lambda_vec, error_train, error_cv] = validationCurve(X, y, Xcv, ycv);
