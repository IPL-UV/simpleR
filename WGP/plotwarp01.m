function plotwarp01(Xw,D,t,colour)

test = [min(t):0.01:max(t)]';
num = (length(Xw) - D - 2)/3; 
for i = 1:num
    ea(i,1) = exp(Xw(D+2+i)); eb(i,1) = exp(Xw(D+2+num+i));
    c(i,1) = Xw(D+2+2*num+i);
end
figure(1),
plot(test,warp(test,ea,eb,c),colour)
hold on,