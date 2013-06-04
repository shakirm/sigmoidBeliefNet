-- Implement mean field variational SBN 
-- Shakir, May 2013

require 'createAdjMatrix'
require 'util'
require 'Estep'
require 'Mstep'
require 'lowerBound'

function variationalSBN(X, netStruct, options)
	
	-- Output variables
	stats = {}; params = {}; post = {};

	-- Get dimensions
	local N = X:size(1); -- data dims
	local D = X:size(2);
	
	local nLayers = netStruct.nLayers;
	local nNodes = netStruct.nNodes;
	local nDims = netStruct.nDims;
	local weightMask = netStruct.weightMatrix;
	local meanMask = netStruct.meanMask;

	-- Algorithm settings
	local maxIter = options.numIter;
	local stepSize = options.stepSize;
	local estepReps = options.estepReps;

	-- Initialise parameters
	local logLik = torch.Tensor(1,maxIter):zero();
	local logLik_n = torch.Tensor(1,N);
	local tmpLik = 0;
	
	--local mu = torch.Tensor(N,nNodes):zero(); -- posterior params	
	--local xi = torch.Tensor(N,nNodes):fill(0.5); -- variational params - should be NxnNodes?
	local mu = torch.rand(N,nNodes); -- posterior params
	local xi = torch.rand(N,nNodes); -- variational params
	
	-- local W = torch.cmul(torch.Tensor(nNodes, nNodes):fill(0.1), weightMask); -- weight matrix
	-- local b = torch.Tensor(1, nNodes):fill(0.5); -- biases
	local W = torch.cmul(torch.randn(nNodes, nNodes), weightMask); -- weight matrix
	local b = torch.randn(1, nNodes); -- biases
	
	local phi = torch.Tensor(1,nNodes):zero(); -- used in computing K
	local K = torch.Tensor(nNodes, nNodes):zero(); --  intermediate 	
	local gradxi = torch.Tensor(1,nNodes):zero(); -- grad vectors
		
	-- EM algorithm
	--xi_n = xi[{{1},{}}];  -- maybe move inside later on.
	for i = 1, maxIter do
		local gradW = torch.Tensor(nNodes,nNodes):zero();
		local gradB = torch.Tensor(1,nNodes):zero();
		--xi_n = torch.Tensor(1,nNodes):fill(0.5);
		
		-- get indices of permuted data
		if true == options.randomiseObs then
			idx = randperm(N); -- something not right when I do this - fix.
		else
			idx = torch.range(1,N); -- no permutation
		end;
		
		-- E-step over all data
		for r = 1, N do
			n = idx[r];
			mu_n = torch.Tensor(1,nNodes):zero();
			xi_n = torch.Tensor(1,nNodes):zero();
			--mu_n = mu[{{n},{}}]:clone();
			--xi_n = xi[{{n},{}}]:clone();
						
			mu_n[{{1},{1,D}}] = X[{{n},{}}]:clone(); -- data set to coord of mu 
			x_n = X[{{n},{}}];
						
			for q = 1,estepReps do
				mu_n, xi_n, phi, phiA, phiB = Estep(mu_n, xi_n, W, b, x_n, options,i); -- update xi_n, mu_n
			end;
			
			--mu[{{n},{}}] = mu_n[{{1},{D+1,nNodes}}]:clone();
			mu[{{n},{}}] = mu_n[{{1},{}}]:clone();
			xi[{{n},{}}] = xi_n:clone();
			
			-- Compute (neg) gradients (accumulate over all obs)
			gradB = gradB + phi - mu_n; -- eq 35
			
			gprod = torch.mm(torch.cmul(torch.Tensor(1,nNodes):fill(1) - phi,xi_n):t(),mu_n);
			gradW = gradW - torch.cdiv(torch.cmul(gprod,MA),MAA);
			
			gprod = torch.mm(torch.cmul(phi,torch.Tensor(1,nNodes):fill(1) - xi_n):t(),mu_n);
			gradW = gradW + torch.cdiv(torch.cmul(gprod,MB),MBB); -- should this be MBB:t()
			
			gradW = gradW + torch.mm((xi_n - mu_n):t(),mu_n); -- -- eq 34
								
			-- Compute lower bound
			logLik_n[1][n] = lowerBound(mu_n, xi_n, phiA, phiB, W, b, D);
		end;
		logLik[1][i] = (torch.sum(logLik_n));
		
		-- M-step: Update the weights and biases
		W, b = Mstep(W, b, gradW, gradB, stepSize, weightMask)
		
		if true == options.plot then
			gnuplot.plot(torch.squeeze(logLik),'+-')
		end;
		if true == options.display then
			print(string.format('\r Iter %d, LogLik %4.4f', i, logLik[1][i]));
		end;
	end; 
	
	stats.logLik = logLik;
	stats.logLik_n = logLik_n;

	post.mu = mu;
	post.xi = xi_n;
	
	params.weight = W;
	params.bias = b;
	
	return post, params, stats
end; -- function