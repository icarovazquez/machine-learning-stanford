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
% Hint: The max function might come in useful. I  n particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

a1 = [ones(size(X,1),1), X];
fprintf('size of a1 is %f\n', size(a1));

fprintf('size of Theta1 is %f\n', size(Theta1));

z2 = a1 * Theta1';
fprintf('size of z2 is %f\n', size(z2));

z3= sigmoid(z2);

a2 = [ones(size(z3,1),1), z3];
fprintf('size of a2 is %f\n', size(a2));

fprintf('size of Theta2 is %f\n', size(Theta2));

a3 = sigmoid(a2 * Theta2'); 
 
fprintf('size of a3 is %f\n', size(a3));

[v p] = max(a3,[],2);

fprintf('size of p is %f\n', size(p));






% =========================================================================


end
