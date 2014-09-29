%This function performs k-fold CV with a specified number of repeats given
%the name of a linear model. CVMatrix(nModels*nFolds) stores MSE.
%Outputs the best linear model. This can then be used on the final held out
%test set.
function [betaCVX, bestModelCoeff, cvMatrix] = ...
    select_model_using_cv(strModelName, coeffRange, nFolds, nRepeats, sampleTrainX, sampleTrainY)

%search for best coeff value
cvMatrix = zeros(length(coeffRange),nFolds);

for k=1:nRepeats%For each repetition of the CV step, get expected CV matrix 
    %Repetition give robustness in data poor conditions

    foldLabels = get_fold_labels(length(sampleTrainX(:,1)),nFolds);%changes for each k due to randomness

    for i=1:length(coeffRange) %For each model
        for j=1:nFolds %For each fold
            clear XEverythingElse XFold YEverythingElse YFold betaCVX
            XEverythingElse  = sampleTrainX(foldLabels~=j,:);
            XFold            = sampleTrainX(foldLabels==j,:);
            YEverythingElse  = sampleTrainY(foldLabels~=j,:);
            YFold            = sampleTrainY(foldLabels==j,:);

            if    (strcmp(strModelName,'Lasso')==1)
                betaCVX = lasso_regression(XEverythingElse,YEverythingElse,coeffRange(i));
            elseif(strcmp(strModelName,'Ridge')==1)
                betaCVX = ridge_regression(XEverythingElse,YEverythingElse,coeffRange(i));
            elseif(strcmp(strModelName,'SVR')==1)
                betaCVX = support_vector_regression(XEverythingElse,YEverythingElse,coeffRange(i));
            end
            Y_tmp_hat_eval  = XFold*betaCVX;
            cvMatrix(i,j) = cvMatrix(i,j) + ...
                (1/nRepeats)*( Y_tmp_hat_eval-YFold )'*(Y_tmp_hat_eval-YFold )/length(YFold); % MSE
        end
    end
end

%% Final model with the best regularization coefficient

[~,bestModelIndex] = min(sum(cvMatrix,2));%Min for MSE
bestModelCoeff = coeffRange(bestModelIndex);

if    (strcmp(strModelName,'Lasso')==1)
    betaCVX = lasso_regression(sampleTrainX,sampleTrainY,bestModelCoeff);
elseif(strcmp(strModelName,'Ridge')==1)
    betaCVX = ridge_regression(sampleTrainX,sampleTrainY,bestModelCoeff);
elseif(strcmp(strModelName,'SVR')==1)
    betaCVX = support_vector_regression(sampleTrainX,sampleTrainY,bestModelCoeff);
end