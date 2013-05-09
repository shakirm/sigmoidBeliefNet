-- Main file to execute different algorithms for learning in sigmoid belief networks.
-- Shakir, May 2013

require 'getData'
require 'testpdf'
require 'slicesample'

local seed = 1;
torch.manualSeed(seed);

local fname = 'bars';
--local fname = 'blockImages'
--local fname = 'mnist'
local showData = 0;
local options;

if 'bars' == fname then
	local nObs = 100;
	local imSize = 9; -- use 3x3 images
	options  = {nObs, imSize} 
	
elseif 'blockImages' == fname then
	local nObs = 100;
	local imSize = 36; -- use 6x6 images
	options  = {nObs, imSize}
	
elseif 'mnist' == fname then
	options = {};	
	
else
	error('**Error: Unknown data type');
end;

-- Get training data
X = getData(fname,options,showData)












