-- Main file to execute different algorithms for learning in sigmoid belief networks.
-- Shakir, May 2013

require 'getData'
--require 'sampleSBN'
require 'variationalSBN'

local seed = 1;
torch.manualSeed(seed);

fname = 'test'
-- local fname = 'bars';
--local fname = 'blockImages'
--local fname = 'mnist'
local showData = 0;
local options, nObs, imSize;

if 'bars' == fname then
	nObs = 100;
	imSize = 9; -- use 3x3 images
	options  = {nObs, imSize} 
	
elseif 'blockImages' == fname then
	nObs = 100;
	imSize = 36; -- use 6x6 images
	options  = {nObs, imSize}
	
elseif 'mnist' == fname then
	options = {};
	
elseif 'test' == fname then
	nObs = 10;
	imSize = 30;
	options = {nObs, imSize}                                                 
else
	error('**Error: Unknown data type');
end

-- Get training data
X = getData(fname,options,showData);

-- Specify architecture of network
layer1dim = 5; -- layer 1 dim
layer2dim = 5; -- layer 2 dim
netStruct = torch.Tensor({layer2dim, layer1dim, imSize}); -- from top down.

options = {};
options.numIter = 20;
options.stepSize = 1e-3;

netStruct = torch.Tensor({1,3,imSize})
out = variationalSBN(X, netStruct, options);

-- samples = sampleSBN(X, nLayers, nDims)


