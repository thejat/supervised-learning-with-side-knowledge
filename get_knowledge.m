function knowledge = get_knowledge(strKnowledgeType,X,Y)

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
    knowledge.quadratic.LHS = XSorted'*bandMatrix'*bandMatrix*XSorted;
end


if(strcmp(strKnowledgeType,'QuadraticOld'))
    %Sort X and Y
    [YSorted,idxSorted] = sort(Y);
    XSorted = X(idxSorted,:);
    idxClose=zeros(length(YSorted)-1,1);
    for i=1:size(XSorted,1)-1
        if(abs(YSorted(i+1) - YSorted(i))/std(YSorted) < .5)
            idxClose(i) = 1;
        end
        knowledge.quadratic.closeBecause(i) = abs(YSorted(i+1) - YSorted(i))/std(YSorted);
    end
    
    knowledge.quadratic.idxClose = idxClose;
    knowledge.quadratic.YSorted = YSorted;
%             knowledge.quadratic.A(k,:) = XSorted(i,:);
%             knowledge.quadratic.A(k+1,:) = X(i,:);
%             knowledge.quadratic.absDiff(k:k+1,:) = abs(Y(j) - Y(i));%this value is stored in two indices.
%             k=k+2;

    
%     
%     if isfield(knowledge.quadratic,'A')
%         knowledge.quadratic.RHS = 1.1*0.5*(norm(knowledge.quadratic.absDiff,2)^2);%because absDiff has repetitions, there is a 0.5 factor. Factor 1.1 gives some wiggle room.
%         bandMatrix = diag(ones(size(knowledge.quadratic.A,1),1)) - [zeros(size(knowledge.quadratic.A,1)-1,1)  diag(ones(size(knowledge.quadratic.A,1)-1,1)); zeros(1,size(knowledge.quadratic.A,1))];
%         knowledge.quadratic.LHS = knowledge.quadratic.A'*bandMatrix'*bandMatrix*knowledge.quadratic.A;
%     else
%         knowledge.quadratic.LHS = zeros(size(X,2),size(X,2));
%         knowledge.quadratic.RHS = 1;
%     end
end



