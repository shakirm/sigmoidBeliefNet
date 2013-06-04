-- Main file to execute different algorithms for learning in sigmoid belief networks.
-- Shakir, May 2013

require 'getData'
require 'variationalSBN'
require 'ancestralSampleSBN'

local seed = 1;
torch.manualSeed(seed);


local fname = 'bars';
--local fname = 'blockImages'
--local fname = 'mnist'

local showData = 1;
local options, nObs, imSize;

if 'bars' == fname then
	nObs = 50;
	imSize = 9; -- use 3x3 images
	options  = {nObs, imSize} 
	outName = 'barsOut.dat'
	
elseif 'blockImages' == fname then
	nObs = 100;
	imSize = 36; -- use 6x6 images
	options  = {nObs, imSize}
	outName = 'blockOut.dat'
	
elseif 'mnist' == fname then
	options = {};
	
else
	error('**Error: Unknown data type');
end

-- Get training data
X = getData(fname,options,showData);

--gnuplot.imagesc(X[{{1,10},{}}])
--gnuplot.figure(6);
--gnuplot.imagesc(X[{{2},{}}]:resize(3,3));

-- Specify architecture of network
local options = {};
options.numIter = 500;
options.stepSize = 1e-1;
options.estepReps = 20;
options.updateXi = true;
options.stepSizeXi = 5e-3;

-- Algorithm options
options.randomiseObs = false; -- to randomise order in which data points are seen.
options.plot = true; -- plot bound values
options.display = true; -- print on screen
options.saveOut = true; -- save results to disk

-- Architecture of belief network
local netStruct = {}
netStruct.arch = torch.Tensor({2, 3, 5,imSize}) -- architecture from top to bottom
netStruct.nLayers = #netStruct.arch; -- length(netStruct)
netStruct.nNodes = torch.sum(netStruct.arch); 
netStruct.nDims = netStruct.nNodes - netStruct.arch[netStruct.nLayers]; -- total number of latent nodes
netStruct.weightMatrix, netStruct.meanMask = createAdjMatrix(netStruct.arch);

-- -- Inference and Learning
if nil == io.open(outName,'r') then -- if no stored result, then run first
	post, params, stats = variationalSBN(X, netStruct, options);

	if true == options.saveOut then
		res = {post, params, stats}
		torch.save(outName, res)
	end;
else
	res = torch.load(outName);
	post = res[1];
	params = res[2];
	stats = res[3];
end;

-- Analysis
print('Final Marginal Likelihood', stats.logLik[#stats.logLik])

-- Generate samples from model
genOptions = {}
genOptions.nSamples = 10
genOptions.seed = 10;
samples = ancestralSampleSBN(netStruct, params, post, genOptions)


gnuplot.figure(2)
gnuplot.imagesc(samples); -- plot all samples as matrix
gnuplot.figure(3)
gg = samples[{{4},{}}] -- resize one example and look at result
gnuplot.imagesc(gg:resize(3,3))


