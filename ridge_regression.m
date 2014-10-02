function betaCVX = ridge_regression(X,Y,C,knowledge)

if(strcmp(knowledge.strType,'None'))

    cvx_begin quiet
        variable betaCVX(size(X,2))
        minimize( ( X*betaCVX-Y )'*( X*betaCVX-Y )/length(Y) + C*(betaCVX'*betaCVX) )
    cvx_end

end

if(strcmp(knowledge.strType,'Linear'))

    cvx_begin quiet
        variable betaCVX(size(X,2))
        minimize( ( X*betaCVX-Y )'*( X*betaCVX-Y )/length(Y) + C*(betaCVX'*betaCVX) )
        subject to
           knowledge.linear.A*betaCVX <= 0
    cvx_end

end


if(strcmp(knowledge.strType,'Quadratic'))

    cvx_begin quiet
        variable betaCVX(size(X,2))
        minimize( ( X*betaCVX-Y )'*( X*betaCVX-Y )/length(Y) + C*(betaCVX'*betaCVX) )
        subject to
           betaCVX'*knowledge.quadratic.LHS*betaCVX <= knowledge.quadratic.RHS
    cvx_end

end


if(strcmp(knowledge.strType,'Conic'))

    cvx_begin quiet
        variable betaCVX(size(X,2))
        minimize( ( X*betaCVX-Y )'*( X*betaCVX-Y )/length(Y) + C*(betaCVX'*betaCVX) )
        subject to
            for i=1:knowledge.conic.m
               knowledge.conic.X(i,:)*betaCVX + knowledge.conic.r*norm(betaCVX,2) <= knowledge.conic.Y(i) 
            end
    cvx_end

end