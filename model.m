clear
% Load the dataset
load("dataset.mat")
Xdataset = x;
Ydataset = y;

%Splitting training and testing data on a random basis
splitPercentage = 0.75;
randomIndex = randperm(length(x));

xTrain = x(randomIndex(1:splitPercentage*length(x)),:);
yTrain = y(randomIndex(1:splitPercentage*length(x)),:);

xTest = x(randomIndex(round(splitPercentage*length(x))+1:end),:);
yTest = y(randomIndex(round(splitPercentage*length(x))+1:end),:);

%The vectors to store the  errs calculated for changing values  of lambda
errTrainingVector = [];
errTestingVector  = [];
errRegTrainingVector = [];
errRegTestingVector = [];


upperLambda=1000;

for i = 0:upperLambda
    [err,errReg,model,errT,errRegT] = multivariateRegression(xTrain,yTrain,i,xTest,yTest);
    errTrainingVector= [errTrainingVector ; err];
    errTestingVector = [errTestingVector  ; errT];
    errRegTrainingVector = [errRegTrainingVector ; errReg];
    errRegTestingVector = [errRegTestingVector ; errRegT];
end

%Plotting all the relevant  graph for  validatio
lambdaMatrix = (0:upperLambda);
inverseLambdaMatrix = 1./lambdaMatrix;

[minimumErrTesting, optimalLambdaIndex] = min(errTestingVector);
[minimumErrRegTesting, optimalRegLambdaIndex] = min(errTestingVector);

clf
figure("Name","Empirical Error CrossValidation Against Lambda");
plot(lambdaMatrix, errTrainingVector','red')
hold on
plot(lambdaMatrix,errTestingVector','blue')
xline(optimalLambdaIndex,'green')
xlabel('Lambda')
ylabel('Empirical Error Risk')
legend('Training Error','Testing Error','Optimal Lambda')
actualOptimalLambda = optimalLambdaIndex - 1;

figure("Name","Empirical Error CrossValidation Against 1/Lambda");
plot(inverseLambdaMatrix , errTrainingVector','red')
hold on
plot(inverseLambdaMatrix ,errTestingVector','blue')
xline(1/optimalLambdaIndex,'green')
xlabel('1/Lambda')
ylabel('Empirical Error Risk')
legend('Training Error','Testing Error','Optimal 1/Lambda')

figure("Name","Regularized Error CrossValidation Against Lambda");
plot(lambdaMatrix, errRegTrainingVector','red')
hold on
plot(lambdaMatrix,errRegTestingVector','blue')
xline(optimalRegLambdaIndex,'green')
xlabel('Lambda')
ylabel('Regularized Error Risk')
legend('Training Error','Testing Error','Optimal Lambda')
actualOptimalRegLambda = optimalRegLambdaIndex - 1;

figure("Name","Regularized Error CrossValidation Against 1/Lambda");
plot(inverseLambdaMatrix , errRegTrainingVector','red')
hold on
plot(inverseLambdaMatrix ,errRegTestingVector','blue')
xline(1/optimalRegLambdaIndex,'green')
xlabel('1/Lambda')
ylabel('Regularized Error Risk')
legend('Training Error','Testing Error','Optimal 1/Lambda')

disp('Using Optimal Lambda from Empirical Risk Plot')
actualOptimalLambda
[err, Rerr,Rmodel,errT,RerrT] = multivariateRegression(xTrain,yTrain,optimalLambdaIndex,xTest,yTest)

disp('Using Optimal Lambda from Regularized  Risk Plot')
actualOptimalRegLambda
[err, Rerr,Rmodel,errT,RerrT] = multivariateRegression(xTrain,yTrain,optimalRegLambdaIndex,xTest,yTest)

%To plot N dimensional data, we are going to use mesh. Here the axis going
%from 1 to 100, represent each feature. The ones going from 1-400
%represents the dataset. and the height represents the value of that being
%plotted.
%In this model, we fit our model by multiplying the initial values of the
%dataset, with the Rmodel(theta) values, to get predictions of the value
%for each point. That is what we have plot over the mesh data in the black
%colour. It can considered a model at each point where the 1-100 axis is
%a new dimension. We could  have seen the whole thing in 100 2d graphs
%also. This is just one way to repesent multivariate data.
figure("Name","Data visualisation of N dimension");

mesh(Xdataset)
hold on
temp = repmat(Rmodel,400);
temp = temp(1:100,:);
mesh(temp','edgecolor', 'k');
xlabel('Feature number : x1,x2...x100')
ylabel('Data point')
zlabel('Value')
