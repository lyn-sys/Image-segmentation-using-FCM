function [C, dist, J] = fcm(X, k, b)
iter = 0;
[N, p] = size(X);
P = randn(N, k);

P = P./(sum(P, 2)*ones(1, k));
J_prev = inf; 
J = [];
while true,iter = iter + 1;
    t = P.^b;
    C = (X'*t)'./(sum(t)'*ones(1, p));
    dist = sum(X.*X, 2)*ones(1, k) + (sum(C.*C, 2)*ones(1, N))'-2*X*C';
    t2 = (1./dist).^(1/(b-1));
    P = t2./(sum(t2, 2)*ones(1, k));
    J_cur = sum(sum((P.^b).*dist))/N;
    J = [J J_cur];

    if norm(J_cur-J_prev, 'fro') < 1e-3, break;
    end

    fprintf('#iteration: %03d, objective function: %f\n', iter, J_cur);
    J_prev = J_cur;
end