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


