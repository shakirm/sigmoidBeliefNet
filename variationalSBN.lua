-- Implement mean field variational SBN 
-- Shakir, May 2013

require 'createAdjMatrix'
require 'util'

function variationalSBN(X, netStruct, params)

	-- Get dimensions
	local nLayers = #netStruct; -- length(netStruct)
	local N = X:size(1); -- data dims
	local D = X:size(2);
	local nNodes = torch.sum(netStruct); 
	local nDims = nNodes - netStruct[nLayers]; -- total number of latent nodes

	-- Algorithm settings
	local maxIter = params.numIter;
	local stepSize = params.stepSize;

	-- Gradients that will be used in computation
	local weightMask, meanMask = createAdjMatrix(netStruct);
	
	---------------------- DEBUGGING ---------------------
	-- for testing i'll just use a upper triangular matrix
	nNodes = 10;
	nDims = 5;
	weightMask = torch.Tensor(nNodes,nNodes):fill(1);
	weightMask = torch.triu(weightMask,1)
	meanMask = torch.Tensor(1,nNodes):zero();
	meanMask[{{1},{1,nDims}}] = 1;
	
	-- THIS FOR DEBUGGING ONLY
	local W = torch.cmul(torch.Tensor(nNodes, nNodes):fill(0.1), weightMask); -- weight matrix
	local b = torch.Tensor(1, nNodes):fill(0.5); -- biases
	------------------------------------------------------

	-- Initialise parameters
	local logLik = torch.Tensor(1,maxIter):zero();
	local logLik_n = torch.Tensor(1,N);
	local tmpLik = 0;
	local mu = torch.Tensor(N,nNodes):zero(); -- posterior params
	local xi = torch.Tensor(N,nNodes):fill(0.5); -- variational params
	--local W = torch.cmul(torch.rand(nNodes, nNodes), weightMask);
-- -- weight matrix
--	local b = torch.rand(1, nNodes); -- biases
	
	local phi = torch.Tensor(1,nNodes):zero(); -- used in computing K
	local K = torch.Tensor(nNodes, nNodes):zero(); --  intermediate 
	
	local gradxi = torch.Tensor(1,nNodes):zero(); -- grad vectors
	local gradW = torch.Tensor(nNodes,nNodes):zero();
	local gradB = torch.Tensor(1,nNodes):zero();
			
	-- EM algorithm
	for i = 1, maxIter do	
		-- E-step over all data
		
		for n = 1, N do
			-- Do we need to do the update a few times for each data point - ADD later ????		
			-- Compute K matrix and update mu_n
			mu_n = mu[{{n},{}}];
			mu_n[{{1},{1,D}}] = X[{{n},{}}]:clone(); -- data set to coord of mu 
			xi_n = xi[{{n},{}}];
			
			-- compute phi
			MA0 = torch.exp(-torch.cmul(b,xi_n)); -- 1xN
			MA = torch.exp(-torch.cmul(W,torch.repeatTensor(xi_n,nNodes,1))); -- NxN
			MAA = torch.repeatTensor(torch.Tensor(1,nNodes):fill(1) - mu_n,nNodes,1) + torch.cmul(MA,torch.repeatTensor(mu_n,nNodes,1));
			
			phiA = torch.cmul(MA0, torch.prod(MAA,2)); -- eq 30
			
			MB0 = torch.exp(torch.cmul(b,torch.Tensor(1,nNodes):fill(1)-xi_n));
			MB = torch.exp(torch.cmul(W,torch.repeatTensor(torch.Tensor(1,nNodes):fill(1) - xi_n,nNodes,1))); -- NxN	
			MBB = torch.repeatTensor(torch.Tensor(1,nNodes):fill(1) - mu_n,nNodes,1) + torch.cmul(MB,torch.repeatTensor(mu_n,nNodes,1));
			
			phiB = torch.cmul(MB0, torch.prod(MBB,2)); -- eq 31
		
			phi = torch.cdiv(phiB,phiA + phiB); -- eq 32 --CORRECT TILL 
			
			-- Compute intermediate matrix K
			negPhi = torch.repeatTensor(torch.Tensor(1,nNodes):fill(1) - phi,nNodes,1);
			negMA = torch.Tensor(nNodes,nNodes):fill(1) - MA;
			
			KA = torch.cdiv(torch.cmul(negPhi,negMA:t()):t(),MAA); --K0
			
			negMB = torch.Tensor(nNodes,nNodes):fill(1) - MB;
			KB = torch.cdiv(torch.cmul(torch.repeatTensor(phi,nNodes,1),negMB:t()):t(),MBB)
			
			K = KA + KB; -- eq 33 -- ok till here
			
			-- Gradient of xi
			xiprodA = torch.cmul(torch.repeatTensor(mu_n,nNodes,1),torch.cmul(W,MA))
			gradXiA = -torch.cmul((b+torch.sum(torch.cdiv(xiprodA,MAA),2)):t(),phiA);
		
			xiprodB = torch.cmul(torch.repeatTensor(mu_n,nNodes,1),(torch.cmul(W,MB)))
			gradXiB = -torch.cmul(b+torch.sum(torch.cdiv(xiprodB,MBB),2):t(),phiB);
			
			gradXi = torch.cdiv(gradXiA + gradXiB, phiA + phiB) + b + torch.mm(W,mu_n:t()):t();
			
			-- Update mu_n	
			mu_n = sigmoid(b + torch.mm(W,mu_n:t()) + torch.mm(W:t(),(mu_n - xi_n):t()) + torch.sum(K,1)) -- eq 24, dim (1 x nNode)
			mu_n[{{1},{1,D}}] = X[{{n},{}}]:clone(); -- data set to coord of mu 
			
			-- Compute gradients (accumulate over all obs)
			gradB = gradB + phi - mu_n; -- eq 35
			
			gprod = torch.mm(torch.cmul(torch.Tensor(1,nNodes):fill(1) - phi,xi_n):t(),mu_n)
			gradW = gradW - torch.cdiv(torch.cmul(gprod,MA),MAA);
			
			gprod = torch.mm(torch.cmul(phi,torch.Tensor(1,nNodes):fill(1) - xi_n):t(),mu_n)
			gradW = gradW + torch.cdiv(torch.cmul(gprod,MB),MBB);
			
			gradW = gradW + torch.mm((xi_n - mu_n):t(),mu_n); -- -- eq 34
		
			
			-- Compute lower bound
			_mu = mu_n[{{},{D+1,nNodes}}]
			
			negMu = torch.Tensor(_mu:size()):fill(1) - _mu;
			KL = torch.sum(torch.cmul(_mu,torch.log(_mu)) + torch.cmul(negMu,torch.log(negMu)))  -- KL term
			expect1 = -torch.sum(torch.log(phiA + phiB));
			expect2 = torch.squeeze(torch.mm((mu_n - xi_n),torch.mm(W,mu_n:t())+ b));
			logLik_n[1][n] = -(expect1 + expect2 + KL);			
		end;
		logLik[1][i] = (torch.sum(logLik_n));
		
		-- M-step: Update the weights and biases
		W = W - torch.mul(gradW, stepSize); -- can this be replaced with other opimiser?
		W = torch.cmul(W,weightMask);
		b = b - torch.mul(gradB,stepSize);
		gnuplot.plot(torch.squeeze(logLik),'+-')
	end;
	print(logLik)
end; -- function

