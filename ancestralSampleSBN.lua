-- Generate images from a sigmoid belief network
-- Shakir, May 2013
require 'util'

function ancestralSampleSBN(netStruct, params, post, options)
	
	torch.manualSeed(options.seed);
	
	local nSamples = options.nSamples;
	local D = netStruct.nDims;
	local L = netStruct.nNodes;
	local botSt, botEn, topSt, topEn;
	local b, wight, prob, samp;
	local samples = torch.Tensor(nSamples, netStruct.arch[netStruct.nLayers[1]]);
	
	for s = 1, nSamples do
		-- top layer
		botSt = 1; -- staring idx for lower layer
		botEn = netStruct.arch[1]; -- last idx for lower layer
		
		b = params.bias[{{1},{botSt,botEn}}];
		
		prob = sigmoid(b)
		samp = sampleBernoulli(prob)
		
		-- All subsequent layers
		for i = 2,netStruct.nLayers[1] do 
			sz = netStruct.arch[i];
			topSt = botSt;
			topEn = botEn;
			botSt = topEn + 1;
			botEn = topEn + sz;
			
			b = params.bias[{{1},{botSt,botEn}}]
			weight = params.weight[{{topSt,topEn},{botSt, botEn}}]
			
			prob = sigmoid(torch.mm(samp:double(),weight) + b);
			samp = sampleBernoulli(prob)
		end;
		--samples[{{s},{}}] = samp:clone();
		samples[{{s},{}}] = prob:clone(); -- at end return probs instead
	end;
	
	return samples;
end; -- function

function sampleBernoulli(prob)
	local D = prob:size(2);
	local u = torch.rand(1,D);
	local bitVec = u:lt(prob);
	
	return bitVec;
end -- sampleBernoulli