%Generate multiple random fullsamples (training, test, side knowledge) with
%the same coefficients and run regression with side knowledge while learning.

%% Step 1: Generate simulated data

clc;close all;clear all;

s = RandStream('mcg16807','Seed',0) ; RandStream.setGlobalStream(s);
nRuns       = 1;
nBeta       = 10; %fixed dimension. Higher means, more data is required.
betaTrue    = randn(nBeta,1);
nTrainArray = floor(nBeta*[3.1]);
nTest       = mean(nTrainArray);%can be anything.
nKnowledge  = 100;
noiseSigma  = 1;

%The features need not be random here. Only the residuals need to be random.

% featureSigma = randn(nBeta); %+1 for covariance with residuals.
% featureSigma = featureSigma'*featureSigma;
featureSigma = gallery('randcorr',nBeta);
sampleTestX = randn(nTest,nBeta)*chol(featureSigma); %kept same for a given beta
sampleTestY = sampleTestX*betaTrue + noiseSigma*randn(nTest,1);
sampleKnowledgeX = randn(nKnowledge,nBeta)*chol(featureSigma); %kept same for a given beta
sampleKnowledgeY = sampleKnowledgeX*betaTrue; %No need to add noise% + noiseSigma*randn(nKnowledge,1);

for i=1:nRuns
    sampleTrainX = randn(max(nTrainArray),nBeta)*chol(featureSigma);
    sampleTrainY = sampleTrainX*betaTrue + noiseSigma*randn(max(nTrainArray),1);
end

%% Step 2a: Ordinary least squares

betaOLS = sampleTrainX\sampleTrainY;
figure(1); plot(betaOLS,betaTrue,'.'); title(['norm(betaOLS-betaTrue): ' num2str(norm(betaOLS-betaTrue,2))])

%% Step 2: Fit model without side Knowledge: Ridge Regression

nFolds   = 5;
nRepeats = 1;
coeffRange = 2.^([-7:1:0]);
[betaBaseline, bestModelCoeff, cvMatrix] = ...
    select_model_using_cv('Ridge', coeffRange, nFolds, nRepeats, sampleTrainX, sampleTrainY, zeros(1,nBeta));

%%
figure(2); plot(betaBaseline,betaTrue,'.'); title(['norm(betaBaseline-betaTrue): ' num2str(norm(betaBaseline-betaTrue,2))])

%metrics = metric_of_success(sampleTestY,Y_hat_val,length(sampleTrainX(1,:)),'Val',str_dependent,'Ridge',plot_enable);


%% Step 2b: Fit model with side knowledge: 




%{

plot of performance (measured using RMSE) as a function of training sample
size.

For st plot performance for without knowledge case with error bars.

%}
