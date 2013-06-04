
require 'util'

h = randperm(5);
print(h)
for n = 1, h:size(1) do
	print(h[n])
end;


hh = torch.rand(3,3)
hh = torch.Tensor({{1,2},{math.nan, 9},{3,4}})
print(hh)
hh[{{3},{3}}] = math.nan
print(isnan(hh))


-------------------- DEBUGGING ---------------------
	-- for testing i'll just use a upper triangular matrix
	--nNodes = 10;
--	nDims = 5;
--	weightMask = torch.Tensor(nNodes,nNodes):fill(1);
--	weightMask = torch.triu(weightMask,1)
--	meanMask = torch.Tensor(1,nNodes):zero();
--	meanMask[{{1},{1,nDims}}] = 1;

--	-- THIS FOR DEBUGGING ONLY
	--local W = torch.cmul(torch.Tensor(nNodes, nNodes):fill(0.1), weightMask); -- weight matrix
--	local b = torch.Tensor(1, nNodes):fill(0.5); -- biases
--	------------------------------------------------------


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
