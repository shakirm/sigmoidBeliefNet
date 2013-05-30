-- Main file to execute different algorithms for learning in sigmoid belief networks.
-- Shakir, May 2013

require 'getData'
require 'variationalSBN'

local seed = 1;
torch.manualSeed(seed);

--fname = 'test'
local fname = 'bars';
-- local fname = 'blockImages'
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
	imSize = 5;
	options = {nObs, imSize}                                                 
else
	error('**Error: Unknown data type');
end

-- Get training data
X = getData(fname,options,showData);

-- Specify architecture of network
local options = {};
options.numIter = 500;
options.stepSize = 1e-2;
options.estepReps = 10;
options.updateXi = true;
options.stepSizeXi = 1e-3;

-- Architecture of belief network
local netStruct = {}
netStruct.arch = torch.Tensor({5, 7,imSize})
netStruct.nLayers = #netStruct.arch; -- length(netStruct)
netStruct.nNodes = torch.sum(netStruct.arch); 
netStruct.nDims = netStruct.nNodes - netStruct.arch[netStruct.nLayers]; -- total number of latent nodes
netStruct.weightMatrix, netStruct.meanMask = createAdjMatrix(netStruct.arch);

---------------------- Only for debgging ------------
-- for test case with fully cconnected DAG
--local netStruct = {};
--netStruct.nLayers = 3; -- length(netStruct)
--netStruct.nNodes = imSize + 5; -- use 10 hidden unites in a dag structure 
--netStruct.nDims = 5;
--netStruct.weightMatrix = torch.triu(torch.Tensor(netStruct.nNodes,netStruct.nNodes):fill(1)) -- total number of latent nodes
--idx = torch.eye(netStruct.nNodes):byte();
--netStruct.weightMatrix[idx] = 0;
--netStruct.meanMask = torch.Tensor(1,netStruct.nNodes):fill(1);
--netStruct.meanMask[{{1},{1, imSize}}] = 0;
-----------------------------------------------------

-- Inference and Learning
out = variationalSBN(X, netStruct, options);


