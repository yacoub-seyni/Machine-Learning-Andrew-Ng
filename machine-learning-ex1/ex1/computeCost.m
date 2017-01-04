function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

for i = 1:m
    % fprintf('theta: [%f, %f]\n', size(transpose(theta)));
    % fprintf('X: [%f, %f]\n\n', size(X(i,:)));
    % fprintf('y: [%f, %f]\n\n', size(y(i,:)));
    % fprintf('J: [%f, %f]', size(theta' .* X(i,:)));

    J = J + ( ( dot(transpose(theta), X(i,:)) - y(i,:) ).^2 );
    
end

J = J / (2 * m);

% =========================================================================

end
