function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0; grad = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
val = 0; hold = 0;
a1 = 0; a2 = 0; a3 = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
for i = 1:m
        a1 = [ones(1, 1) X(i,:)];
        hold = sigmoid(Theta1 * a1');
        a2 = [ones(1, 1); hold];
        a3 = sigmoid(a2' * Theta2');
    for k = 1:size(a3, 2)
    
        if (y(i,1) == k) 
            val = 1;
        else 
            val = 0;
        end
        
        J = J + ( -val * log(a3(1,k)) - (1 - val) * log(1 - a3(1,k)) );
    end
end

J = J / m;

accum = 0;

for j = 1:size(Theta1, 1)
   for k = 2:size(Theta1, 2)
      accum = accum + ( Theta1(j,k) .* Theta1(j,k) ); 
   end
end

for j = 1:size(Theta2, 1)
   for k = 2:size(Theta2, 2)
       accum = accum + ( Theta2(j,k) .* Theta2(j,k) );
   end
end

accum = lambda * accum / (2 * m);

J = J + accum;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

act_1 = 0; act_2 = 0; act_3 = 0;
delta_1 = 0; delta_2 = zeros(26,26);
delta_3 = zeros(1,num_labels);
z_2 = 0;

for t = 1:m
    act_1 = [ones(1,1) X(t,:)];
    act_2 = [ones(1,1); sigmoid(Theta1 * act_1')];
    act_3 = sigmoid(act_2' * Theta2');
    
    for i = 1:num_labels
        if i == y(t)
            delta_3(i) = act_3(i) - 1;
        else
            delta_3(i) = act_3(i);
        end
    end

    z_2 = sigmoidGradient(Theta1 * act_1');
    z_2 = [ones(1,1); z_2];
    
    delta_2 = ( delta_3 * Theta2) .* z_2';
    delta_2 = delta_2(2:end);

    Theta1_grad = Theta1_grad + (delta_2' * act_1);
    Theta2_grad = Theta2_grad + (delta_3' * act_2');
    
end

Theta1_grad = Theta1_grad ./ m;
Theta2_grad = Theta2_grad ./ m; 

for i = 1:size(Theta1_grad, 1)
   for j = 2:size(Theta1_grad, 2)
      Theta1_grad(i,j) = Theta1_grad(i,j) + (lambda / m) .* Theta1(i,j);
   end
end

for i = 1:size(Theta2_grad, 1)
   for j = 2:size(Theta2_grad, 2)
      Theta2_grad(i,j) = Theta2_grad(i,j) + (lambda / m) .* Theta2(i,j);
   end
end


grad = [Theta1_grad(:); Theta2_grad(:)];

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
% grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
