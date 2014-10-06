%Generate multiple random fullsamples (training, test, side knowledge) with
%the same coefficients and run regression with side knowledge while learning.

% Step 1: Generate simulated data
clc;close all;clear all;
if matlabpool('size') > 0, matlabpool close, end
matlabpool('open', 'local', 4)

s = RandStream('mcg16807','Seed',999) ; RandStream.setGlobalStream(s);
nRuns       = 30;
nBeta       = 60; %fixed dimension. Higher means, more data is required.
betaTrue    = randn(nBeta,1);
nTrainArray = floor(nBeta*5*[1.5:.5:3]);
nTest       = max(nTrainArray);%can be anything.
nKnowledge  = 2*nBeta; %floor(sqrt(max(nTrainArray)));
noiseSigma  = .005*sqrt(nBeta);

%The features need not be random here. Only the residuals need to be random.

% featureSigma = randn(nBeta); %+1 for covariance with residuals.
% featureSigma = featureSigma'*featureSigma;
featureSigma = gallery('randcorr',nBeta);
sampleTestX = randn(nTest,nBeta)*chol(featureSigma); %kept same for a given beta
sampleTestY = sampleTestX*betaTrue + noiseSigma*randn(nTest,1) + noiseSigma*trnd(1,nTest,1);
sampleKnowledgeX = randn(nKnowledge,nBeta)*chol(featureSigma); %kept same for a given beta
sampleKnowledgeY = sampleKnowledgeX*betaTrue; %No need to add noise% + noiseSigma*randn(nKnowledge,1);



tic %For console logging.
for i=1:nRuns %Multiple samples from the data source.
    
    sampleTrainX = randn(max(nTrainArray),nBeta)*chol(featureSigma);
    sampleTrainY = sampleTrainX*betaTrue + noiseSigma*randn(max(nTrainArray),1) + noiseSigma*trnd(1,max(nTrainArray),1);

    % Step 2: In this step, we will fit three models, OLS, ridge regression and
    % ridge regression with side knowledge.
    
    %Some common settings
    nFolds   = 5;
    nRepeats = 1;
    coeffRange = 2.^([-7:2:0]);

    % Step 2a: Ordinary least squares (without any regularization)

    for j=1:length(nTrainArray)
        betaOLS{j} = sampleTrainX(1:nTrainArray(j),:)\sampleTrainY(1:nTrainArray(j));
        metricsOLS{j} = metric_of_success(sampleTestY,sampleTestX*betaOLS{j},size(sampleTestX,2),'Test','Ridge',0);
    end


    % Step 2b: Ridge regression without side knowledge (with ell_2 regularization)
    
    knowledgeNone = get_knowledge('None',sampleKnowledgeX,sampleKnowledgeY,betaTrue);
    parfor j=1:length(nTrainArray)
        fprintf('Run %d of %d: Without knowledge: CV for sample size index j = %d of %d.\n', i, nRuns, j, length(nTrainArray));
        [betaBaseline{j}, bestBaselineCoeff{j}, cvBaselineMatrix{j}] = ...
            select_model_using_cv('Ridge', coeffRange, nFolds, nRepeats, sampleTrainX(1:nTrainArray(j),:), sampleTrainY(1:nTrainArray(j)), knowledgeNone);
        metricsBaseline{j} = metric_of_success(sampleTestY,sampleTestX*betaBaseline{j},size(sampleTestX,2),'Test','Ridge',0);
    end
    toc
    
    % Step 2c: Ridge regression with Linear side knowledge (with ell_2 regularization)

    knowledgeLinear = get_knowledge('Linear',sampleKnowledgeX,sampleKnowledgeY,betaTrue);
    parfor j=1:length(nTrainArray)
        fprintf('Run %d of %d: With Linear knowledge: CV for sample size index j = %d of %d.\n', i, nRuns, j, length(nTrainArray));
        [betaLinear{j}, bestLinearCoeff{j}, cvLinearMatrix{j}] = ...
            select_model_using_cv('Ridge', coeffRange, nFolds, nRepeats, sampleTrainX(1:nTrainArray(j),:), sampleTrainY(1:nTrainArray(j)), knowledgeLinear);
        metricsLinear{j} = metric_of_success(sampleTestY,sampleTestX*betaLinear{j},size(sampleTestX,2),'Test','Ridge',0);
    end
    toc
    
    % Step 2d: Ridge regression with Quadratic side knowledge (with ell_2 regularization)

    knowledgeQuadratic = get_knowledge('Quadratic',sampleKnowledgeX,sampleKnowledgeY,betaTrue);
    parfor j=1:length(nTrainArray)
        fprintf('Run %d of %d: With Quadratic knowledge: CV for sample size index j = %d of %d.\n', i, nRuns, j, length(nTrainArray));
        [betaQuadratic{j}, bestQuadraticCoeff{j}, cvQuadraticMatrix{j}] = ...
            select_model_using_cv('Ridge', coeffRange, nFolds, nRepeats, sampleTrainX(1:nTrainArray(j),:), sampleTrainY(1:nTrainArray(j)), knowledgeQuadratic);
        metricsQuadratic{j} = metric_of_success(sampleTestY,sampleTestX*betaQuadratic{j},size(sampleTestX,2),'Test','Ridge',0);
    end
    toc
    
    % Step 2e: Ridge regression with Conic side knowledge (with ell_2 regularization)

    knowledgeConic = get_knowledge('Conic',sampleKnowledgeX,sampleKnowledgeY,betaTrue);
    parfor j=1:length(nTrainArray)
        fprintf('Run %d of %d: With Conic knowledge: CV for sample size index j = %d of %d.\n', i, nRuns, j, length(nTrainArray));
        [betaConic{j}, bestConicCoeff{j}, cvConicMatrix{j}] = ...
            select_model_using_cv('Ridge', coeffRange, nFolds, nRepeats, sampleTrainX(1:nTrainArray(j),:), sampleTrainY(1:nTrainArray(j)), knowledgeConic);
        metricsConic{j} = metric_of_success(sampleTestY,sampleTestX*betaConic{j},size(sampleTestX,2),'Test','Ridge',0);
    end
    toc

    % Collecting performance statistics
    for j=1:length(nTrainArray)
        rmseOLSArray(j) = metricsOLS{j}.rmse;
        rmseBaselineArray(j) = metricsBaseline{j}.rmse;
        rmseLinearArray(j) = metricsLinear{j}.rmse;
        rmseQuadraticArray(j) = metricsQuadratic{j}.rmse;
        rmseConicArray(j) = metricsConic{j}.rmse;

    end
    runDataOLS(:,i) = rmseOLSArray'; 
    runDataBaseline(:,i) = rmseBaselineArray';
    runDataLinear(:,i) = rmseLinearArray';
    runDataQuadratic(:,i) = rmseQuadraticArray';
    runDataConic(:,i) = rmseConicArray';
end

matlabpool close;

% Plotting
runDataOLSMean = mean(runDataOLS,2); runDataOLSStd = std(runDataOLS,1,2);
runDataBaselineMean = mean(runDataBaseline,2); runDataBaselineStd = std(runDataBaseline,1,2);
runDataLinearMean = mean(runDataLinear,2); runDataLinearStd = std(runDataLinear,1,2);
runDataQuadraticMean = mean(runDataQuadratic,2); runDataQuadraticStd = std(runDataQuadratic,1,2);
runDataConicMean = mean(runDataConic,2); runDataConicStd = std(runDataConic,1,2);


y = [runDataOLSMean runDataBaselineMean runDataLinearMean runDataQuadraticMean runDataConicMean];% nrows is sample size, ncols is methods
errY = [runDataOLSStd runDataBaselineStd runDataLinearStd runDataQuadraticStd runDataConicStd];
figure(1); h = barwitherr(errY, y);% Plot with errorbars
set(gca,'XTickLabel',nTrainArray);
legend('Multiple Linear Regression','Ridge Regression','With Linear','With Quadratic', 'With Conic');
ylabel('RMSE (lower is better)');
set(h(1),'FaceColor','k');
xlabel('Sample size');

% y = [runDataBaselineMean runDataLinearMean runDataQuadraticMean runDataConicMean];% nrows is sample size, ncols is methods
% errY = [runDataBaselineStd runDataLinearStd runDataQuadraticStd runDataConicStd];
% figure(1); h = barwitherr(errY, y);% Plot with errorbars
% set(gca,'XTickLabel',nTrainArray);
% legend('Ridge Regression','With Linear','With Quadratic', 'With Conic');
% ylabel('RMSE (lower is better)');
% set(h(1),'FaceColor','k');
% xlabel('Sample size');