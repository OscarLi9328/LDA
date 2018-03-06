% Predicting the Fisher discriminant by using the KNN-classifier

function [ y_hat ] = classify_Fisher( projected_test_data, projected_train_data, train_labels, k )

M_test = length(projected_test_data);
M_train = length(projected_train_data);
knearest = zeros(M_test, k);

for i = 1:M_test
    for j = 1:M_train
        dist(i,j) = norm(projected_train_data(j,:) -  projected_test_data(i,:));
    end
    [~ , ind] = sort(dist(i,:) );
    knearest(i,:) = ind(1:k)';
    j_ind = knearest(i,:);
    class(i) = mode(train_labels(j_ind));    

end

y_hat = class';

end

