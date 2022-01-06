function [err,model,errT] = ridgereg(x,y,lambda,xT,yT)
% x = vector of input scalars for training
% y = vector of output scalars for training
% lambda = the penalty parameter
% xT = vector of input scalars for testing
% yT = vector of output scalars for testing
% err = average squared loss on training
% model = vector of polynomial parameter coefficients
% errT = average squared loss on testing

xtx = x' * x;
model =(x' * x + lambda * eye(size(xtx))) \ x' * y;
err = (1/(2*size(x,1)))*sum((y-x*model).^2);
if (nargin==5)
    errT = (1/(2*size(xT,1))) * sum((yT - xT*model).^2);

end
