function [m] = metric_of_success(Y_truth,Y_hat,nfeat,str_insample,str_method,plot_enable)

str_dependent = 'Value';

m.rmse = norm(Y_hat-Y_truth)/sqrt(length(Y_hat));
m.rsq = 1 - ((norm(Y_truth-Y_hat,2)^2)/(norm(Y_truth-mean(Y_truth),2)^2));
m.adjrsq = 1-(1-m.rsq)*(length(Y_truth)-1)/(length(Y_truth)-nfeat);
        
if (plot_enable==1)        
    figure;
    scatter([Y_truth],[Y_hat]);
    title([str_insample ':' str_method ':(rmse,rsq,adjrsq)=(' num2str(m.rmse) ',' num2str(m.rsq) ',' num2str(m.adjrsq) ')'],'FontSize',18);
    xlabel(['True ' str_dependent],'FontSize',18);
    ylabel(['Estimated ' str_dependent],'FontSize',18);
    hold on; tempx = min(Y_truth):0.005:max(Y_truth); tempy = tempx; plot(tempx,tempy,'g'); hold off; %45 degree line
end
%if error greater in test than in train, then this could be because of
%overfitting: variance.
%if both errors are high, this could be underfitting: bias
% R2 describes the proportion of variance of the dependent variable explained by the regression model. 
% If the regression model is ?perfect?, SSE is zero, and R2 is 1. 
% If the regression model is a total failure, SSE is equal to SST, no variance is explained by regression, and R2 is zero. 
% It is important to keep in mind that there is no direct relationship between high R2 and causation.