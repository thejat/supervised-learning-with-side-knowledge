function knowledge = get_knowledge(strKnowledgeType,X,Y,betaTrue)

knowledge.strType = strKnowledgeType;

if(strcmp(strKnowledgeType,'None'))
    return;
end

if(strcmp(strKnowledgeType,'Linear'))
    k=1;
    for i=1:size(X,1)
        for j=1:size(X,1)
            if(Y(j) < Y(i))
                knowledge.linear.A(k,:) = X(j,:) - X(i,:);
            else
                knowledge.linear.A(k,:) = X(i,:) - X(j,:);
            end
            k=k+1;
        end
        if(k>10*size(X,1))
            break;
        end
    end
end


if(strcmp(strKnowledgeType,'Quadratic'))
    %Sort X and Y
    [YSorted,idxSorted] = sort(Y);
    XSorted = X(idxSorted,:);
    for i=1:size(XSorted,1)-1
        knowledge.quadratic.absDiff(i) = abs(YSorted(i+1) - YSorted(i));
    end
    
    knowledge.quadratic.RHS = 1.1*(norm(knowledge.quadratic.absDiff,2)^2);%Factor 1.1 gives some wiggle room.
    bandMatrix = diag(ones(size(XSorted,1),1));
    bandMatrix = bandMatrix(1:end-1,:);
    bandMatrix = bandMatrix - [zeros(size(XSorted,1)-1,1)  diag(ones(size(XSorted,1)-1,1))];
    knowledge.quadratic.LHS = XSorted'*(bandMatrix'*bandMatrix)*XSorted;
end

if(strcmp(strKnowledgeType,'Conic'))
    knowledge.conic.r = 0.1;
    knowledge.conic.m = size(X,1);
    knowledge.conic.X = X;
    knowledge.conic.Y = 1.1*(Y + knowledge.conic.r*norm(betaTrue,2));
end

%% Debugging
% X = sampleTrainX; Y = sampleTrainY;
% C = .0001;
% A{1} = ones(1,size(X,2));
% A{2} = zeros(1,size(X,2));
% cvx_begin quiet
%     variable betaCVX(size(X,2))
%     minimize( ( X*betaCVX-Y )'*( X*betaCVX-Y )/length(Y) + C*betaCVX'*betaCVX )
%     subject to
%         for i=1:2
%            A{i}*betaCVX + norm(betaCVX,2) <= 6 
%         end
% cvx_end
% cvx_optval
