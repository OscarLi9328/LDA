% Establish the linear discriminant function
% Output:
%       w    weight function of size (class, N)
%       w_0  bias function of size (class, 1)

function [x_proj, w] = Fisher(featureVector, labels)
% load the dataset and seperate the training data from the dataset

x = featureVector';
[M, N] = size(featureVector);
C = length(countcats(labels));

num = zeros(1,C);
m = zeros(N,C);
Sw = zeros(N);
Sb = zeros(N);

% preprocess the data
for class = 1:C
    num(class) = sum(double(labels) == class);
end

% computing the mean of each class\
for class = 1:C
    mu(class,:) = mean(featureVector((double(labels) == class),:));
end
mu = mu';
mu_total = mean(featureVector)'; % mean of the whole dataset

% compute the within-class covariance
for class = 1:C
    Sb = Sb + num(class) * (mu(:,class) - mu_total) * (mu(:,class) - mu_total)';
end
% Sb = (mu - mu_total*ones(1,C)) * (mu - mu_total*ones(1,C))';

% computing the within-class covariance
for class = 1:C
    for j = 1:M
        if (double(labels(j)) == class)
            Sw = Sw + ((x(:,j) - mu(:,class)) * (x(:,j) - mu(:,class))');
        end
    end
end

% the eigenvector and the eigenvalue
[eigenVec, eigenVal] = eig(Sw\Sb);

% Determine the eigenvectors corresponding to the largest eigenvalues
[~,idx] = sort(diag(real(eigenVal)), 'descend');
w = real(eigenVec(:,idx(1:C-1)));

x_proj = w'* x; % project the data 

end
