function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%


% we transpose theta so we can get do the (X * theta) multiplication
% we then substract Y and square the result
% we element-wise multiply by R so we only get the values where a user rated a movie
% we add the whole thing and then multiply by 1/2
% we could have gotten the whole thing in 1 line of code but had to break it up
% because we need to use some of the intermediate terms for other calculations


predicted_ratings = X * Theta';
rating_error = (predicted_ratings - Y).^2;

error_factor = rating_error .* R;

J = (1/2) * sum(sum(error_factor));

% computing unregularized gradients
% had to recalculate the intermediate terms since the gradients don't need to be squared

the_rating_error = predicted_ratings -Y;
the_error_factor = the_rating_error .* R;

X_grad = the_error_factor * Theta;
Theta_grad = the_error_factor' * X;



% computing regularized cost function
reg_term = lambda/2 * sum(sum(Theta.^2)) + lambda/2 * sum(sum(X.^2));

J = J + reg_term;


% calculating regularized gradients

X_grad = X_grad + lambda * X;
Theta_grad = Theta_grad +  lambda * Theta;







% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
