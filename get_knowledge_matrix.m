function knowledgeMatrix = get_knowledge_matrix(strKnowledgeType,X,Y)

if(strcmp(strKnowledgeType,'None'))
    knowledgeMatrix = zeros(1,size(X,2));
end

if(strcmp(strKnowledgeType,'Linear'))
    for i=1:size(X,1)
        for j=1:size(X,1)
            if(Y(j) < Y(i))
                knowledgeMatrix(i,:) = X(j,:) - X(i,:);
            else
                knowledgeMatrix(i,:) = X(i,:) - X(j,:);
            end
        end
    end
end
