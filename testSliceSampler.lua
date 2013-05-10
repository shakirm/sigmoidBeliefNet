
require 'testpdf'
require 'slicesample'

seed = 1;
torch.manualSeed(seed)

x = torch.range(-5,5,0.01);
hh = torch.exp(testpdf(x, params));
hh2 = torch.Tensor(hh:size(1),1);
hh2[{{},{1}}] = hh;
	
plot = 1;
if 1==plot then
local params = {};
	
	gnuplot.figure(1)
	gnuplot.plot(x,hh,'-')
end;
	
nSamples = 7000;
initVec = torch.rand(1,1);
width = 3;
stepOut = 1;
samples = slicesample(nSamples, testpdf, initVec, width, stepOut, params)

-- Save results

function save_csv(fname, matrix)
	local f = io.open(fname,'w')
	if matrix:dim() ~= 2 then error ('2D matrices only') end
	for i=1,matrix:size(1) do
		local str = table.concat(matrix:select(1,i):clone():storage():totable(),',')
		f:write(str .. '\n')
	end
	f:close()
end

save_csv('samples.csv',samples)
save_csv('true.csv',hh2);

gnuplot.figure(2)
gnuplot.plot(torch.squeeze(samples[{{},{1}}]))
gnuplot.figure(3)
gnuplot.hist(torch.squeeze(samples[{{5000,samples:size(1)},{1}}]))
	