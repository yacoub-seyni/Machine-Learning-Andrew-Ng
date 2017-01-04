function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

A1 = [ones(m, 1) X];

fprintf('A1: [%f, %f]\n', size(A1));
fprintf('theta1: [%f, %f]\n', size(Theta1));
% fprintf('max: [%f, %f]\n', size(max((X * all_theta'), [], 2)));


Z2 = Theta1 * A1';
A2 = sigmoid(Z2);
fprintf('A2: [%f, %f]\n', size(A2));
A2 = A2';
fprintf('A2: [%f, %f]\n', size(A2));
[m0, n0] = size(A2);

fprintf('size: %f\n', m0);
A2 = [ones(m0, 1) A2];
    
fprintf('Z2: [%f, %f]\n', size(Z2));
fprintf('A2: [%f, %f]\n', size(A2));
fprintf('theta2: [%f, %f]\n', size(Theta2));

Z3 = Theta2 * A2';
A3 = sigmoid(Z3);

fprintf('Z3: [%f, %f]\n', size(Z3));
fprintf('A3: [%f, %f]\n', size(A3));

[p, i] = max(A3, [], 1);
p = i;
p = p';
fprintf('p: [%f, %f]\n', size(p));

% =========================================================================


end
