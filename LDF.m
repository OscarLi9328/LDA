% Establish the linear discriminant function

function [w, x_tilt] = LDF(featureVector, label)
% load the dataset and seperate the training data from the dataset
x = featureVector';

[M, N] = size(featureVector);
C = length(unique(label)); % number of classes
x_tilt = [ones(M,1), x']; % pad one column on train_data
T = zeros(M,C);
% 
% for i = 1:M
%     if (double(label(i)) == 1)
%         T(i, 1) = 30/66;
%     else
%         T(i, 2) = 36/66;
%     end
% end



for class = 1:C
    for i = 1:M
        if (double(label(i))) == class
            T(i, class) = 1;
        end
    end 
end

w_tilt = (x_tilt'*x_tilt) \( x_tilt' * T);
w = w_tilt;

end
