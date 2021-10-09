function [err, errReg,model,errT, errRegT] = polyreg_problem2(x,y,lambda,xT,yT)
% Finds a Regularized error over a multivariate data
%
%    function [err,model,errT] = polyreg(x,y,D,xT,yT)
%
% x = matrix of input multivariate features
% y = vector of output scalars for training
% lambda = factor for penalty
% xT = vector of input scalars for testing
% yT = vector of output scalars for testing
% err = empirical average squared loss on training
% errReg = average Regularized loss on training
% model = vector of polynomial parameter coefficients
% errT = average squared loss on testing
% errRegT = average Regularized loss on testing
%
% Example Usage:
%
% x = multivariate data point
% lambda = penalty 
% y = output values
% [err,model] = polyreg_problem2(x,y,4);

[rows cols] = size(x);

%model calculation with lambda factor for penalty
model = inv(x'*x + eye(cols)*lambda) * x'*y;

%empirical error on the training
err   = ((1/(2*length(x)))*sum((y-x*model).^2));

%regularized error on the training(empirical Error + penalty)
errReg = err + ((lambda/(2*length(x)))*sum((model.^2)));

if (nargin==5)  
  
  %empirical error on the testing
  errT  = (1/(2*length(xT)))*sum((yT-xT*model).^2);
  
  %regularized error on the testing(empirical Error + penalty)
  errRegT = errT + ((lambda/(2*length(xT)))*sum((model.^2)));

end

%plotting is difficult for multivariate, we cannot plot various feature in
%the same graph, so plotting code in in the problem2.m file instead of here
%because we  will plot it once