function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

m = size(X,1);

distance_matrix = zeros(m, K);


for i = 1:K
   temp_var = bsxfun(@minus,X,centroids(i,:));
   distance_matrix(:,i) = sum(temp_var.^2,2);
   
   %[v p] = min(distance_matrix(i,:));
   %idx(i) = p;

endfor


% we can do everything in one for loop. Need to come back to fix it
for i = 1:m

% find the row with the closest centroid
[v p] = min(distance_matrix(i,:));

% assign the row number to the idx vector
idx(i) = p;
  
endfor


% =============================================================

end

