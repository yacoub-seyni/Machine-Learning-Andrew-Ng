function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

n = size(theta);
sum_cost = 0;
sum_grad = zeros(n);

for i = 1:m
    sum_cost = sum_cost + ( -y(i,1) * log( sigmoid(theta' * X(i,:)') ) ...
        - (1 - y(i,1)) * ( log(1 - sigmoid(theta' * X(i,:)'))) );   
end

% temp = theta;
% temp(1) = 0;
% 
% J = -y' * log( sigmoid(X * theta)) - (1 - y)' * log(1 - sigmoid(X * theta));
% J = J / m;

% fprintf('size 1: %f\n', size(theta));
% fprintf('size 2: %f\n', size(X(i,:)));
% fprintf('size 3: %f\n', size(theta' * X(i,:)'));

for j = 1:n
    for i = 1:m
        sum_grad(j) = sum_grad(j) ...
            + ( sigmoid(theta' * X(i,:)') - y(i,1) ) * (X(i,j));
    end
end

% grad = (X' * (sigmoid(X * theta) - y));
% grad = grad ./ m;

J = sum_cost / m;
grad = sum_grad ./ m;

% fprintf('size 4: %f', size(grad));

% =============================================================

end
