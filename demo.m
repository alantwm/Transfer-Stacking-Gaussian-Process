clear
clc

n_train = 20;
n_test = 100;
d=8;

%% Gen Data
% Generates data from related problems of DTLZ1b variants, normalizes the data, 
% and creates source models.
[ytest,ytrain,xtest,xtrain,src_models] = gen_data(n_train,n_test,d);

%% Build TSGP Model
model_tsgp = tsgp(xtrain,ytrain,src_models);

%% Test
ytest_hat_tsgp = model_tsgp.predict(xtest);
rmse_tsgp = sqrt(mse(ytest_hat_tsgp - ytest));

model_gp = fitrgp(xtrain,ytrain,'KernelFunction','ardsquaredexponential');
ytest_hat_gp = model_gp.predict(xtest);
rmse_gp = sqrt(mse(ytest_hat_gp - ytest));

rmse_tsgp
rmse_gp
