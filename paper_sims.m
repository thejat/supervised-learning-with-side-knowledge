%Generate multiple random fullsamples (training, test, side knowledge) with
%the same coefficients and run regression with side knowledge while learning.

% Step 1: Generate simulated data

clc;close all;clear all;

s = RandStream('mcg16807','Seed',1000) ; RandStream.setGlobalStream(s);
nRuns       = 4;
nBeta       = 20; %fixed dimension. Higher means, more data is required.
betaTrue    = randn(nBeta,1);
nTrainArray = floor(nBeta*[1.5:1:8.5]);
nTest       = max(nTrainArray);%can be anything.
nKnowledge  = nBeta; %floor(sqrt(max(nTrainArray)));
noiseSigma  = 0.3*sqrt(nBeta);

%The features need not be random here. Only the residuals need to be random.

% featureSigma = randn(nBeta); %+1 for covariance with residuals.
% featureSigma = featureSigma'*featureSigma;
featureSigma = gallery('randcorr',nBeta);
sampleTestX = randn(nTest,nBeta)*chol(featureSigma); %kept same for a given beta
sampleTestY = sampleTestX*betaTrue + noiseSigma*randn(nTest,1);
sampleKnowledgeX = randn(nKnowledge,nBeta)*chol(featureSigma); %kept same for a given beta
sampleKnowledgeY = sampleKnowledgeX*betaTrue; %No need to add noise% + noiseSigma*randn(nKnowledge,1);


tic
for i=1:nRuns %Multiple samples from the data source.

    sampleTrainX = randn(max(nTrainArray),nBeta)*chol(featureSigma);
    sampleTrainY = sampleTrainX*betaTrue + noiseSigma*randn(max(nTrainArray),1);

    % Step 2: In this step, we will fit three models, OLS, ridge regression and
    % ridge regression with side knowledge.
    
    %Some common settings
    nFolds   = 5;
    nRepeats = 1;
    coeffRange = 2.^([-7:2:0]);

    % Step 2a: Ordinary least squares (without any regularization)

    for j=1:length(nTrainArray)

        betaOLS{j} = sampleTrainX(1:nTrainArray(j),:)\sampleTrainY(1:nTrainArray(j));
        %figure(1); plot(betaOLS,betaTrue,'.'); title(['norm(betaOLS-betaTrue): ' num2str(norm(betaOLS-betaTrue,2))])
        metricsOLS{j} = metric_of_success(sampleTestY,sampleTestX*betaOLS{j},size(sampleTestX,2),'Test','Ridge',0);
    end


    % Step 2b: Ridge regression without side knowledge (with ell_2 regularization)

    % The following are the parameter settings we will use for this section.

    knowledgeNone = get_knowledge('None',sampleKnowledgeX,sampleKnowledgeY);
    if(matlabpool('size') == 0) matlabpool; end
    parfor j=1:length(nTrainArray)
        fprintf('Run %d of %d: Ridge Reg. without knowledge: CV for sample size index j = %d of %d.\n', i, nRuns, j, length(nTrainArray));
        [betaBaseline{j}, bestBaselineCoeff{j}, cvBaselineMatrix{j}] = ...
            select_model_using_cv('Ridge', coeffRange, nFolds, nRepeats, sampleTrainX(1:nTrainArray(j),:), sampleTrainY(1:nTrainArray(j)), knowledgeNone);
        %figure(2); plot(betaBaseline,betaTrue,'.'); title(['norm(betaBaseline-betaTrue): ' num2str(norm(betaBaseline-betaTrue,2))])
        metricsBaseline{j} = metric_of_success(sampleTestY,sampleTestX*betaBaseline{j},size(sampleTestX,2),'Test','Ridge',0);
    end
    toc

            
    % Step 2c: Ridge regression with linear side knowledge (with ell_2 regularization)

    % The following are the parameter settings we will use for this section.

    knowledgeLinear = get_knowledge('Linear',sampleKnowledgeX,sampleKnowledgeY);

    parfor j=1:length(nTrainArray)
        fprintf('Run %d of %d: Ridge Reg. with linear knowledge: CV for sample size index j = %d of %d.\n', i, nRuns, j, length(nTrainArray));
        [betaLinear{j}, bestLinearCoeff{j}, cvLinearMatrix{j}] = ...
            select_model_using_cv('Ridge', coeffRange, nFolds, nRepeats, sampleTrainX(1:nTrainArray(j),:), sampleTrainY(1:nTrainArray(j)), knowledgeLinear);
        %figure(3); plot(betaLinear,betaTrue,'.'); title(['norm(betaLinear-betaTrue): ' num2str(norm(betaLinear-betaTrue,2))])
        metricsLinear{j} = metric_of_success(sampleTestY,sampleTestX*betaLinear{j},size(sampleTestX,2),'Test','Ridge',0);
    end
    toc

    knowledgeQuadratic = get_knowledge('Quadratic',sampleKnowledgeX,sampleKnowledgeY);

    parfor j=1:length(nTrainArray)
        fprintf('Run %d of %d: Ridge Reg. with quadratic knowledge: CV for sample size index j = %d of %d.\n', i, nRuns, j, length(nTrainArray));
        [betaQuadratic{j}, bestQuadraticCoeff{j}, cvQuadraticMatrix{j}] = ...
            select_model_using_cv('Ridge', coeffRange, nFolds, nRepeats, sampleTrainX(1:nTrainArray(j),:), sampleTrainY(1:nTrainArray(j)), knowledgeQuadratic);
        %figure(3); plot(betaQuadratic,betaTrue,'.'); title(['norm(betaQuadratic-betaTrue): ' num2str(norm(betaQuadratic-betaTrue,2))])
        metricsQuadratic{j} = metric_of_success(sampleTestY,sampleTestX*betaQuadratic{j},size(sampleTestX,2),'Test','Ridge',0);
    end
    toc
    
    
    % Collecting performance statistics
    for j=1:length(nTrainArray)
        rmseOLSArray(j) = metricsOLS{j}.rmse;
        rmseBaselineArray(j) = metricsBaseline{j}.rmse;
        rmseLinearArray(j) = metricsLinear{j}.rmse;
        rmseQuadraticArray(j) = metricsQuadratic{j}.rmse;
    end
    runDataOLS(:,i) = rmseOLSArray'; 
    runDataBaseline(:,i) = rmseBaselineArray';
    runDataLinear(:,i) = rmseLinearArray';
    runDataQuadratic(:,i) = rmseQuadraticArray';
end

% Plotting
runDataOLSMean = mean(runDataOLS,2); runDataOLSStd = std(runDataOLS,1,2);
runDataBaselineMean = mean(runDataBaseline,2); runDataBaselineStd = std(runDataBaseline,1,2);
runDataLinearMean = mean(runDataLinear,2); runDataLinearStd = std(runDataLinear,1,2);
runDataQuadraticMean = mean(runDataQuadratic,2); runDataQuadraticStd = std(runDataQuadratic,1,2);

figure(3); plot(nTrainArray,runDataOLSMean,'b-'); hold on;
plot(nTrainArray,runDataOLSMean - runDataOLSStd,'b--'); 
plot(nTrainArray,runDataOLSMean + runDataOLSStd,'b--'); 
plot(nTrainArray,runDataBaselineMean,'r-');
plot(nTrainArray,runDataBaselineMean - runDataBaselineStd,'r--'); 
plot(nTrainArray,runDataBaselineMean + runDataBaselineStd,'r--');
plot(nTrainArray,runDataLinearMean,'g-');
plot(nTrainArray,runDataLinearMean - runDataLinearStd,'g--'); 
plot(nTrainArray,runDataLinearMean + runDataLinearStd,'g--'); 
plot(nTrainArray,runDataQuadraticMean,'k-');
plot(nTrainArray,runDataQuadraticMean - runDataQuadraticStd,'k--'); 
plot(nTrainArray,runDataQuadraticMean + runDataQuadraticStd,'k--'); 
hold off;
title('RMSE of various methods')
ylabel('RMSE (lower is better)')
xlabel('Sample size')


