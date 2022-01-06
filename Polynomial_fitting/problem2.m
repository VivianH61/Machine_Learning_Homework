load('problem2.mat');

% two-fold cross validation
index = crossvalind('Kfold',400,2);
x1 = x (index == 1,:);
y1 = y (index == 1);
x2 = x (index == 2,:);
y2 = y (index == 2);

err_train = [];
err_test = [];
models = {};


lambdas = 0:0.5:1000;
for i = 1:length(lambdas)
    [err,model,errT] = ridgereg(x1,y1,lambdas(i),x2,y2);
    err_train(i) = err;
    err_test(i) = errT;
    models{i} = model;
end
close all;
figure
h1 = plot(lambdas, err_train,'color',[1 0 0]);M1 = "Training Risk";
hold on;
h2 = plot(lambdas, err_test,'color',[0 1 0]);M2 = "Testing risk";
legend([h1,h2], [M1, M2]);
xlabel('lambda');
ylabel('Risk');
hold off;