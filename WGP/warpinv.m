function t = warpinv(z,ea,eb,c,t,N)

% provides a numerical inverse to the warp function using Newton
% iterations
%
% N: number of iterations
% ea,eb,c: warp parameters

for n = 1:N
  t = t - (warp(t,ea,eb,c) - z)./dwarp(t,ea,eb,c);
end

function df = dwarp(t,ea,eb,c)

df = ones(size(t));
for i = 1:length(ea)
     df = df + ea(i)*eb(i)*(1 - (tanh(eb(i)*(t + c(i)))).^2);
end