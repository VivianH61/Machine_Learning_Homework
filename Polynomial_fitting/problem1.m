
function main
X = load('problem1.mat', 'x');
Y = load('problem1.mat', 'y');
x = cell2mat(table2array(cell2table(struct2cell(X))));
y = cell2mat(table2array(cell2table(struct2cell(Y))));


D_value = [1:30];
err_wholeset = [];
models = {};
for i = 1 : length(D_value)
    [err, model] = polyreg(x,y,i);
    err_wholeset(i) = err;
    models{i} = model;
    % when D == 11, the err is the min
    % disp("D = " + i + ", err = " + err);
end


% figure
% plot(D_value,err_wholeset);
% title('empitical measure of model with different degree of the polynominal')
% xlabel('D (degree of the polynominal)')
% ylabel('average squared loss on testing')



% two-fold cross validation
D_value = [1:30];
err_value = zeros([1,length(D_value)]);
errT_value = zeros([1,length(D_value)]);
% randomly split the data into two halves
n = length(x);
R = randperm(n);
x1 = x(R(1:n/2));
y1 = y(R(1:n/2));
x2 = x(R(n/2+1:n));
y2 = y(R(n/2+1:n));
for i = 1 : length(D_value)
    [err_value(i),model1,errT_value(i)] = polyreg(x1,y1,D_value(i),x2,y2); 
end




figure
h1 = plot(D_value, err_value,'color',[1 0 0]);M1 = "Training Risk";
hold on;
h2 = plot(D_value, errT_value,'color',[0 1 0]);M2 = "Testing risk";
legend([h1,h2], [M1, M2]);
xlabel('D (degree of the polynominal)');
ylabel('Risk');
hold off;

% plot the f(x, theta) with the best choice of D
[min_risk, min_index] = min(errT_value);
polyreg(x1,y1,D_value(min_index),x2,y2);
%}
end