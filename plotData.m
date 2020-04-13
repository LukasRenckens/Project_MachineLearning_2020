function plotData(x, y, xLable)
%PLOTDATA Plots the data points x and y into a new figure 
%   PLOTDATA(x,y) plots the data points 

figure;                               % open a new figure window
plot(x, y, 'rx', 'MarkerSize', 10);   % Plot the data
ylabel('Price in USD');               % Set the y-axis label
xlabel(xLable);                       % Set the x-axis label

end
