function z = warp(t,ea,eb,c)

% warp: provides a non-linear monotonic warping function from a sum of
% tanh functions, where ea and eb are positve parameter vectors

% the number of tanh 'steps' is determined by the length of the
% parameter vectors ea, eb, c

z = t;
for i = 1:length(ea)
     z = z + ea(i)*tanh(eb(i)*(t + c(i)));
end