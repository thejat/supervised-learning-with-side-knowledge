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
nTrainArray = floor(nBeta*5*[1:.5:2.5]);
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
    nRepeats = 3;
    coeffRange = 2.^([-8:2:1]);

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
        
    save experimentData.mat %hackish way of saving state in case of error/interruption
end

matlabpool close;

%% Plotting

dataOLS =  quantile(runDataOLS,[.25 .5 .75],2);
dataBaseline =  quantile(runDataBaseline,[.25 .5 .75],2);
dataLinear =  quantile(runDataLinear,[.25 .5 .75],2);
dataQuadratic =  quantile(runDataQuadratic,[.25 .5 .75],2);
dataConic =  quantile(runDataConic,[.25 .5 .75],2);

y = [dataOLS(:,2) dataBaseline(:,2) dataLinear(:,2) dataQuadratic(:,2) dataConic(:,2)];
errY(:,:,1) = [dataOLS(:,1) dataBaseline(:,1) dataLinear(:,1) dataQuadratic(:,1) dataConic(:,1)];
errY(:,:,2) = [dataOLS(:,3) dataBaseline(:,3) dataLinear(:,3) dataQuadratic(:,3) dataConic(:,3)];
h0 = figure(1); 
width=2;
set(0,'DefaultAxesLineWidth',width);
set(0,'DefaultLineLineWidth',width);
get(0,'Default');
set(gca,'LineWidth',width);
h = barwitherr(errY, y);% Plot with errorbars
set(gca,'XTickLabel',nTrainArray);
legend('Multiple Linear Regression','Ridge Regression','With Linear','With Quadratic', 'With Conic');
ylabel('RMSE (lower is better)','FontSize',22);
set(h(1),'FaceColor','k');
xlabel('Sample size','FontSize',22);
set(gca,'FontSize',18,'fontWeight','bold');
set(findall(h,'type','text'),'fontSize',18,'fontWeight','bold');
% saveas(h0,'experiment.png');
