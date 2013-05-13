-- Script to test slice sampling on a simple 1D function
-- Shakir, May 2013

require 'testpdf'
require 'slicesample'
require 'mycsv'

seed = 1;
torch.manualSeed(seed)
local D, m, s, params, width, stepOut;
	
--local test = 'bivariateGauss';
local test = 'multimodalFn';

-- Specify options for each data set
if test == 'multimodalFn' then
	-- Sample from a 1D multimodal function
	D = 1;
	params = {};
	width = 5;
	stepOut = 1;
	nSamples = 10000;
	burnin = 4000;
	objFn = testpdf
	
elseif test == 'bivariateGauss' then
	-- Sample from a correlated Gaussian
	D = 2;
	m = torch.Tensor(1,D):zero();
	S = torch.Tensor({{1, 0.9},{0.9, 1}});
	params = {mu = m, cov = S};
	width = 1;
	stepOut = 1;
	nSamples = 2000;
	burnin = 500;
	objFn = bivariateGauss;
else
	error('Unknown test');
end;
 	
print(params)

-- Run slice sampler
initVec = torch.rand(1,D);
samples = slicesample(nSamples, objFn, initVec, width, stepOut, params)

-- Save results
save_csv('samples.csv',samples)

-- Some simple plots
if test == 'multimodalFn' then
	-- Plot true function
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

	-- Plot a histogram of the samples
	gnuplot.figure(2)
	gnuplot.plot(torch.squeeze(samples[{{},{1}}]))
	gnuplot.figure(3)
	gnuplot.hist(torch.squeeze(samples[{{burnin,samples:size(1)},{1}}]))
	
elseif test == 'bivariateGauss' then
	gnuplot.figure(2)
	gnuplot.plot(torch.squeeze(samples[{{},{1}}]),torch.squeeze(samples[{{},{2}}]))
	
	gnuplot.figure(3)
	gnuplot.hist(torch.squeeze(samples[{{burnin,samples:size(1)},{1}}]))
	
	gnuplot.figure(4)
	gnuplot.hist(torch.squeeze(samples[{{burnin,samples:size(1)},{2}}]))
end;