
%% Plot samples from slice samples
close all;

f = @(x) exp( -x.^2/2).*(1+(sin(3*x)).^2).*(1+(cos(5*x).^2));        
x = -5:0.001:5;
true = load('true.csv');
samples = load('samples.csv');
area = quad(f,-5,5);
sUse = samples(5000:10:end);
N = size(sUse,1)
[binheight,bincenter] = hist(sUse,50);
h = bar(bincenter,binheight,'hist');
set(h,'facecolor',[0.8 .8 1]);
hold on

hold on
xd = get(gca,'XLim');
xgrid = linspace(xd(1),xd(2),1000);
binwidth = (bincenter(2)-bincenter(1));
y = (N*binwidth/area) * f(xgrid);
plot(xgrid,y,'r','LineWidth',2)
hold off

print -dpdf 'histogram.pdf'