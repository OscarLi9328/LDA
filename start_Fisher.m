%% Project 2 -- Pattern Recognition
% Linear Classification using least square and Fisher discriminant
% Ming-Ju, Li
% This code is to find the linear classifier using the Fisher discriminant

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
        k = 5;
    case 2
        dataset = 'wallpaper';
        k = 15;
    case 3
        dataset = 'taiji';
        k = 35;
end
[train_featureVector, train_labels, test_featureVector, test_labels] = loadDataset(dataset); % load the data
C = length(unique(train_labels));
[M_train, N_train] = size(train_featureVector);
[M_test, N_test] = size(test_featureVector);

[x_train_proj,w] = Fisher(train_featureVector, train_labels);
x_test_proj = (test_featureVector*w)';

% gscatter(x_train_proj(1,:), x_train_proj(2,:));

test_labels_pred = classify_Fisher(x_test_proj',x_train_proj', train_labels, k);

conf_mat = confusionmat(test_labels_pred, test_labels); % confusion matrix to find the accuracy

% normalize the accuracy
for i = 1:C
    for j = 1:C
        effi(i,j) = conf_mat(i,j)/sum(conf_mat(i,:),2);
    end
end

pred_accu = mean(diag(effi));
data_std = std(diag(effi));

% figure();
% visualizeBoundariesFill(1,x_train_proj', train_labels,1, 2, k);
% title('\bf Fisher discriminant using KNN classifier');

%% Extra problem
[w_LDF, ~] = LDF(x_train_proj(:,1:66)', train_labels(1:66));

% create the structure to store the parameters
Coeffs_ex(C,C) = struct();
for i = 1:C-1
    for j = 1:C-1
        if i ~= j
            temp = w_LDF(:,i) - w_LDF(:,j);
            Coeffs_ex(i,j).Const = temp(1);
            Coeffs_ex(i,j).Linear = temp(2:end);
        end
    end
end
MdlLinear_ex = struct('Coeffs', Coeffs_ex);

% plotting
figure();
visualizeBoundaries_mod(MdlLinear_ex ,x_train_proj(:,1:66)', train_labels(1:66), 1, 2);
title('{\bf Extra Problem Fisher}');
