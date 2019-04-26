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
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% forward propagation begins here 

a1 = [ones(size(X,1),1), X];
%fprintf('size of a1 is %f by %f\n', size(a1));


% multiplying with the transpose so we have the right dimensions
z2 = a1 * Theta1';    
%fprintf('size of z2 is %f by %f\n', size(z2));

z21 = sigmoid(z2);
%fprintf('size of z21 is %f by %f\n', size(z21));

a2 = [ones(size(z21,1),1), z21];
%fprintf('size of a2 after adding ones is %f by %f\n', size(a2));

z3 = a2 * Theta2';
%fprintf('size of z3 a is %f by %f\n', size(z3));

a3 = sigmoid(z3);
%fprintf('size of a3 is %f by %f\n', size(a3));

%  now we calculate the unregularized cost function 
y_matrix = eye(num_labels)(y,:);


first_term = -y_matrix .* (log(a3)); 

second_term = (1 - y_matrix) .* (log(1 - a3));

J = (1/m) * sum(sum((first_term - second_term)));

%  calculating the regularized cost function

temp1 = Theta1(:, 2:end);

temp2 = Theta2(:, 2:end);

first_reg_term = sum(sum((temp1).^2)); 

second_reg_term = sum(sum((temp2).^2));

reg_term = (lambda/(2*m)) * (first_reg_term + second_reg_term);

J = J + reg_term;


% calculating backward propagation

% we already have forward propagation implemented so we take a3 from there

% calculating deltas


d3 = a3 - y_matrix;
%fprintf('size of d3 is %f by %f\n', size(d3));


d2 = (d3 * temp2) .* sigmoidGradient(z2);
%fprintf('size of d2 is %f by %f\n', size(d2));

% accumulators

Delta1 = d2' * a1;
%fprintf('size of Delta1 is %f by %f\n', size(Delta1));

Delta2 = d3' * a2;
%fprintf('size of Delta2 is %f by %f\n', size(Delta2));


% unregularized gradients
   
Theta1_grad = (1/m) * Delta1;
%fprintf('size of Theta1_grad is %f by %f\n', size(Theta1_grad));

Theta2_grad = (1/m) * Delta2;
%fprintf('size of Theta2_grad is %f by %f\n', size(Theta2_grad));

% calculte regularized gradients

Theta1(:,1) = 0;
Theta2(:,1) = 0;

modified_Theta1 = (lambda/m) * Theta1;
modified_Theta2 = (lambda/m) * Theta2;

Theta1_grad = Theta1_grad + modified_Theta1;
Theta2_grad = Theta2_grad + modified_Theta2;



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
