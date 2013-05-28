-- Sampling from a sigmoid belief network. 
-- Shakir, May 2013

require 'slicesample'


function sampleSBN(X, nLayers, nDims)

-- Initialise variables
N = X:size(1);
D = X:size(2);

H = {}; W = {};

print(N)
print(nDims)
print(nLayers)

for i = 1, nLayers do
	H[i] = torch.Tensor(N, nDims[i+1]):zero();
	W[i] = torch.Tensor(nDims[i], nDims[i+1]):zero(); 
end;

print(H)
print(W)
os.exit()
	
-- 1. Sample latent variables
	for i = 

-- 2. Sample weights



end;
