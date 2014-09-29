function [Y_hat_val,betaCVX,regularize_coeff,cv_matrix,Y_hat_trn] = ...
    select_model_using_cv(str_model,coeffrange,nfolds,nrepeats,X_trn,Y_trn,X_val,do_not_cv,regularize_coeff)

cv_matrix = [];

if(strcmp(str_model,'RFreg')==1)
    ntrees = coeffrange{1};
    coeffrange = coeffrange{2};
end

if (do_not_cv==0) % is false
        
    %search for best coeff value
    cv_matrix = zeros(length(coeffrange),nfolds);
    
    for k=1:nrepeats%Get expected CV matrix. Robust in data poor conditions
        %k-fold CV: Generating fold labels
        foldLabels = get_fold_labels(length(X_trn(:,1)),nfolds);%changes for each k due to randomness

        for i=1:length(coeffrange)
            for j=1:nfolds
                clear X_tmp X_tmp_eval Y_tmp Y_tmp_eval betaCVX RFmodel Y_tmp_hat_eval
                X_tmp       = X_trn(foldLabels~=j,:);
                X_tmp_eval  = X_trn(foldLabels==j,:);
                Y_tmp       = Y_trn(foldLabels~=j,:);
                Y_tmp_eval  = Y_trn(foldLabels==j,:);

                if(strcmp(str_model,'RFreg')==1)
                    RFmodel = regRF_train(X_tmp(:,1:end-1),Y_tmp,ntrees,coeffrange(i));%{max(floor(length(X_trn(1,:))/3),1)}
                    Y_tmp_hat_eval  = regRF_predict(X_tmp_eval(:,1:end-1),RFmodel);
                else
                    if    (strcmp(str_model,'Lasso')==1)
                        betaCVX = lassoWrapper(X_tmp,Y_tmp,coeffrange(i));
                    elseif(strcmp(str_model,'Ridge')==1)
                        betaCVX = ridge_regression(X_tmp,Y_tmp,coeffrange(i));
                    elseif(strcmp(str_model,'SVR')==1)
                        betaCVX = support_vector_regression(X_tmp,Y_tmp,coeffrange(i));
                    end
                    Y_tmp_hat_eval  = X_tmp_eval*betaCVX;
                end
                cv_matrix(i,j) = cv_matrix(i,j) + (1/nrepeats)*( Y_tmp_hat_eval-Y_tmp_eval )'*(Y_tmp_hat_eval-Y_tmp_eval )/length(Y_tmp_eval); % MSE
            end
        end
    end
    
    [~,best_coeff_index] = min(sum(cv_matrix,2));%Min for RMSE
    regularize_coeff = coeffrange(best_coeff_index);
end
%Final model with the best regularization coefficient
if(strcmp(str_model,'RFreg')==1)
    betaCVX    = regRF_train(X_trn(:,1:end-1),Y_trn,ntrees,regularize_coeff);%{max(floor(length(X_trn(1,:))/3),1)}
    Y_hat_val  = regRF_predict(X_val(:,1:end-1),betaCVX);%overloading: betaCVX is not a linear model
    Y_hat_trn  = regRF_predict(X_trn(:,1:end-1),betaCVX);%overloading: betaCVX is not a linear model
else
    if    (strcmp(str_model,'Lasso')==1)
        betaCVX = lassoWrapper(X_trn,Y_trn,regularize_coeff);
    elseif(strcmp(str_model,'Ridge')==1)
        betaCVX = ridge_regression(X_trn,Y_trn,regularize_coeff);
    elseif(strcmp(str_model,'SVR')==1)
        betaCVX = support_vector_regression(X_trn,Y_trn,regularize_coeff);
    end
    Y_hat_val   = X_val*betaCVX;
    Y_hat_trn   = X_trn*betaCVX;
end