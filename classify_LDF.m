% predict the labels of the input data based on the given weight function

function label = classify_LDF(w, x) 
y = x*w;
[M,N] = size(y);
for i = 1:M
    [~,idx] = max(abs(y(i,:)));
    label(i) = idx;
end
label = label';


end