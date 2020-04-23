clear ; close all;

data_table = readtable('cars_custom.txt');                                 % Read non-randomized data

data_array_cell = table2array(data_table(:,[1:4 7:9 11:14 16 19:28]));     % Convert table to array (cell)                      
data_array_double = table2array(data_table(:, [15 5 6 10 17 18 29]));      % Convert table to array (double)
data_array_cell_2 = num2cell(data_array_double);                           % Convert double to cell

data_array = cat(2, data_array_cell, data_array_cell_2);                   % Combine both cell arrays
data_array = data_array(randperm(size(data_array,1)),:);                   % Randomize order

writecell(data_array, "cars_custom_rand");                                 % Save as .txt

% Column nr. | name
% -------------------------
% 1  | manufacturer_name
% 2  | model_name
% 3  | transmission
% 4  | color
% 5  | engine_fuel
% 6  | engine_has_gas
% 7  | engine_type
% 8  | body_type
% 9  | has_warranty
% 10 | state
% 11 | drivetrain
% 12 | is_exchangeable
% 13 | feature_0
% 14 | feature_1
% 15 | feature_2
% 16 | feature_3
% 17 | feature_4
% 18 | feature_5
% 19 | feature_6
% 20 | feature_7
% 21 | feature_8
% 22 | feature_9
% 23 | price_usd
% 24 | Odometer_value
% 25 | year_produced
% 26 | engine_capacity
% 27 | number_of_photos
% 28 | up_counter
% 29 | duration_listed