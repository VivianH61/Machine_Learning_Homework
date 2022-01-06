n = 5;
psis = cell(n-1, 1);
for i = 1:(n-1)
    psis{i} = rand(2,2);
end
[marginals] = jct(psis);

potentials_T = cell(4,1);
potentials_T{1} = [0.1,0.7;0.8,0.3];
potentials_T{2} = [0.5,0.1;0.1,0.5];
potentials_T{3} = [0.1,0.5;0.5,0.1];
potentials_T{4} = [0.0,0.3;0.1,0.3];
[marginal_T] = jct(potentials_T);

function [marginals] = jct(potentials)
marginals = potentials;
n = size(marginals, 1);
separators = ones(n-1, 2);
% forward
for i = 1:n-2
separators(i,:) = sum(marginals{i});
marginals{i+1} = marginals{i+1}.*(separators(i,:)'*[1,1]);
end

% Backward
for i = 1:n-1
    separators_prev = separators(n-1,:);
    separators(n-i,:) = sum(marginals{n-i+1}, 2)';
    marginals{n-i} = marginals{n-i}.*([1;1]*(separators(n-i,:)./separators_prev));
end
%Normalize
for i = 1:n
    marginals{i} = marginals{i}/sum(sum(marginals{i}));
end

end