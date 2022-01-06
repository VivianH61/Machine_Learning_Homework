function hw3

load dataset.mat

% cross validation
% randomly split the data into two halves
n = size(X,1);
Y(51:100) = -1;
i_train = randsample(n, n/2);
i_test = setdiff(1:n, i_train);

trainX = X(i_train, :);
trainY = Y(i_train, :);

testX = X(i_test, :);
testY = Y(i_train, :);
%{
% linear kernels
[nsv, alpha, bias] = svc(trainX, trainY, 'linear', inf);
output_y = svcoutput(trainX , trainY , testX , 'linear', alpha, bias);
error_linear = svcerror(trainX, trainY, testX, testY, 'linear', alpha, bias);
disp('error linear');
disp(error_linear);

% polynomial kernels
errors = [];
% polynomial kernels with different degree of polynomial
global p1; % p1 is degree of polynomial
for i=1:50
    p1 = i;
    % training
    [nsv, alpha, bias] = svc(trainX, trainY, 'poly', 4);
    % predict
    predictedY = svcoutput(trainX , trainY , testX , 'poly', alpha, bias);
    % test error
    errors(i) = svcerror(trainX, trainY, testX, testY, 'poly', alpha, bias);
end
% plot error with different degree of polynomial
fig1 = figure(1);
plot(1:50, errors);
title('error vs. degrees of polynomial');
print(fig1, '-depsc', 'poly.eps');


%polynomial kernels with different values of C
errors = [];
C = 5:100;
for i = 1:length(C)
    % training model
    [nsv, alpha, bias] = svc(trainX, trainY, 'poly',C(i));
    % test
    predictedY = svcoutput(trainX , trainY , testX , 'poly', alpha, bias);
    % calculate test error
    errors(i) = svcerror(trainX, trainY, testX, testY, 'poly', alpha, bias);
end
% plot error with different values of C
fig2 = figure(2);
plot(C, errors);
title('error vs. C');
print(fig2, '-depsc', 'poly.eps');
    
%}

% RBF kernels with different sigmas
errors = [];
sigmas = 0.1:0.05:2;
for i = 1:numel(sigmas)
    p1 = sigmas(i);
    % training model
    [nsv, alpha, bias] = svc(trainX, trainY, 'rbf', 4);
    % test
    predictedY = svcoutput(trainX , trainY , testX , 'rbf', alpha, bias);
    % calculate test error
    errors(i) = svcerror(trainX, trainY, testX, testY, 'rbf', alpha, bias);
end
% plot performance of RBF kernels with different sigma
fig3 = figure(3);
plot(sigmas, errors);
title('RBF kernels error vs. sigma');
%{
% RBF kernels with different C
errors = [];
C = 0:0.5:5;
for i = 1:length(C)
    % training model
    [nsv, alpha, bias] = svc(trainX, trainY, 'rbf', C(i));
    % test
    predictedY = svcoutput(trainX , trainY , testX , 'rbf', alpha, bias);
    % calculate test error
    errors(i) = svcerror(trainX, trainY, testX, testY, 'rbf', alpha, bias);
end
% plot performance of RBF kernels with different C
fig4 = figure(4);
plot(C, errors);
title('RBF kernels error vs. C');
print(fig4,'-depsc','C.eps');
%}


