function betaCVX = ridge_regression(X,Y,C,knowledge)

if(strcmp(knowledge.strType,'None'))

    cvx_begin quiet
        variable betaCVX(length(X(1,:)))
        minimize( ( X*betaCVX-Y )'*( X*betaCVX-Y )/length(Y) + C*betaCVX'*betaCVX )
    cvx_end

end

if(strcmp(knowledge.strType,'Linear'))

    cvx_begin quiet
        variable betaCVX(length(X(1,:)))
        minimize( ( X*betaCVX-Y )'*( X*betaCVX-Y )/length(Y) + C*betaCVX'*betaCVX )
        subject to
           knowledge.linear.A*betaCVX <= 0
    cvx_end

end