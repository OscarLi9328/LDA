%% Project 2 -- Pattern Recognition
% Linear Classification using least square and Fisher discriminant
% Ming-Ju, Li

% This code is for finding the linear classification using the least square
% approach.
% Process Flow
%           1. Train the model and get the weight function
%           2. Apply the trained model to the test data to predict the
%               classes 
%           3. Calculate the accuracy of the model by normalizing the confusion
%               matrix

clc
clear 
close all

addpath export_fig
% Initiate the environment
% Choose which dataset to use (choices wine, wallpaper, taiji) :
num = input('Enter a number(1 for wine, 2 for wallpaper, 3 for taiji):');
switch num
    case 1
        dataset = 'wine';
    case 2
        dataset = 'wallpaper';
    case 3
        dataset = 'taiji';
end
[train_featureVector, train_labels, test_featureVector, test_labels] = loadDataset(dataset); % load the data
C = length(unique(train_labels)); % find the number of classes
[M_train, N_train] = size(train_featureVector); 
[M_test, N_test] = size(test_featureVector);
% train_labels = grp2idx(train_labels);
% test_labels = grp2idx(test_labels);

[w_LDF, x_tilt] = LDF(train_featureVector, double(train_labels)); % train the weight function

train_tilt = [ones(length(train_featureVector),1), train_featureVector];
pred_train_label = classify_LDF(w_LDF, train_tilt);
conf_mat_train = confusionmat(pred_train_label, double(train_labels));

for i = 1:C
    for j = 1:C
        if sum(conf_mat_train(i,:),2) ~= 0
            effi_train(i,j) = conf_mat_train(i,j)/sum(conf_mat_train(i,:),2);
        end
    end
end

train_accu = mean(diag(effi_train));
train_std = std(diag(effi_train));



test_tilt = [ones(length(test_featureVector),1), test_featureVector]; % add the dummy vector to the first row
pred_test_label = classify_LDF(w_LDF, test_tilt); % predict the labels of the test data

conf_mat = confusionmat(pred_test_label, double(test_labels)); % confusion matrix to find the accuracy

% normalize the accuracy
for i = 1:C
    for j = 1:C
        if sum(conf_mat(i,:),2) ~= 0
            effi(i,j) = conf_mat(i,j)/sum(conf_mat(i,:),2);
        end
    end
end

test_accu = mean(diag(effi));
test_std = std(diag(effi));

% decide which two features to be used
featureA = 2;
featureB = 9;
X_new = [train_featureVector(:,featureA), train_featureVector(:,featureB)]; % getting the new feature vector
W_new = LDF(X_new, train_labels); % get the classifier from the feature vector

% create the structure to store the parameters
Coeffs(C,C) = struct();
for i = 1:C
    for j = 1:C
        if i ~= j
            temp = W_new(:,i) - W_new(:,j);
            Coeffs(i,j).Const = temp(1);
            Coeffs(i,j).Linear = temp(2:end);
        end
    end
end
MdlLinear = struct('Coeffs', Coeffs);

% plotting
figure();
visualizeBoundaries(MdlLinear,test_featureVector, test_labels, featureA, featureB);
title('{\bf Linear Discriminant Classification}');
% export_fig linear_discriminant_example -png -transparent 


%% Extra
W_ex = LDF(X_new(1:66,:), train_labels(1:66));

% create the structure to store the parameters
Coeffs_ex(C,C) = struct();
for i = 1:C-1
    for j = 1:C-1
        if i ~= j
            temp = W_ex(:,i) - W_ex(:,j);
            Coeffs_ex(i,j).Const = temp(1);
            Coeffs_ex(i,j).Linear = temp(2:end);
        end
    end
end
MdlLinear_ex = struct('Coeffs', Coeffs_ex);

% plotting
figure();
visualizeBoundaries_mod(MdlLinear_ex ,X_new(1:66,:), train_labels(1:66), 1, 2);
title('{\bf Extra Problem}');


