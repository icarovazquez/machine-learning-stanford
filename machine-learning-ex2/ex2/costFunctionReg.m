function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

h = sigmoid(X * theta);

theta(1) = 0;

redTerm = -y' * log(h);
blueTerm =  (1-y)' * log(1-h);
 
finalTerm = redTerm - blueTerm;

unregularizedTerm = (1/m) * finalTerm;  % this is the unregularized cost function

thetaSum = theta' * theta;

regularizedTerm = (lambda/(2*m)) * thetaSum; % this is the regularized term

J = unregularizedTerm + regularizedTerm;


%now we do the gradient 

unregGrad =  (1/m) * (X' * (h - y)); 
 

regGrad  = (lambda/m) * theta;

grad = unregGrad + regGrad;





% =============================================================

end
