
require 'testpdf'
require 'slicesample'

seed = 1;
torch.manualSeed(seed)

	plot = 0;
	if 1==plot then
	local params = {};
	x = torch.range(-5,5,0.01);
	hh = testpdf(x, params);
	gnuplot.plot(x,torch.squeeze(hh),'-')
	end;
	
	nSamples = 200;
	initVec = torch.rand(1,1);
	width = 1;
	stepOut = 1;
	samples = slicesample(nSamples, testpdf, initVec, width, stepOut, params)
	
	print(samples)
	


	
	