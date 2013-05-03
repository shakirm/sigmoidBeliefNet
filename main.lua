-- Main file to execute different algorithms for learning in sigmoid belief networks.
-- Shakir, May 2013

require 'getData'

local fname = 'bars';
local showData = 1;
local options;

if 'bars' == fname then
	local nObs = 100;
	local imSize = 3;
	options  = {nObs, imSize}
	
elseif 'blocks' == fname then
	options = {};
	
elseif 'mnist' == fname then
	options = {};	
	
else
	error('Unknown data type');
end;

x,y = getData(fname,options,showData)
