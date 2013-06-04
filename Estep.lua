-- E-step updates for variational sigmoid belief network
-- Compute K matrix and update mu_n
-- Shakir, May 013

function Estep(mu_n, xi_n, W, b, x_n, params,i)
	
	local nNodes = mu_n:size(2);
	
	-- compute phi
	MA0 = torch.exp(-torch.cmul(b,xi_n)); -- 1xN
	MA = torch.exp(-torch.cmul(W,torch.repeatTensor(xi_n,nNodes,1))); -- NxN
	MAA = torch.repeatTensor(torch.Tensor(1,nNodes):fill(1) - mu_n,nNodes,1) + torch.cmul(MA,torch.repeatTensor(mu_n,nNodes,1));
	
	MB0 = torch.exp(torch.cmul(b,torch.Tensor(1,nNodes):fill(1)-xi_n));
	MB = torch.exp(torch.cmul(W,torch.repeatTensor(torch.Tensor(1,nNodes):fill(1) - xi_n,nNodes,1))); -- NxN	
	MBB = torch.repeatTensor(torch.Tensor(1,nNodes):fill(1) - mu_n,nNodes,1) + torch.cmul(MB,torch.repeatTensor(mu_n,nNodes,1));
	
	phiA = torch.cmul(MA0, torch.prod(MAA,2)); -- eq 30
	phiB = torch.cmul(MB0, torch.prod(MBB,2)); -- eq 31
	phi = torch.cdiv(phiB,phiA + phiB); -- eq 32 
	
	-- Compute intermediate matrix K
	negPhi = torch.repeatTensor(torch.Tensor(1,nNodes):fill(1) - phi,nNodes,1);
	negMA = torch.Tensor(nNodes,nNodes):fill(1) - MA;
	negMB = torch.Tensor(nNodes,nNodes):fill(1) - MB;
	
	KA = torch.cdiv(torch.cmul(negPhi,negMA:t()):t(),MAA); --K0
	KB = torch.cdiv(torch.cmul(torch.repeatTensor(phi,nNodes,1),negMB:t()):t(),MBB)
	
	K = KA + KB; -- eq 33 -- ok till here
	
	if true == params.updateXi then
		-- Gradient of xi
		xiprodA = torch.cmul(torch.repeatTensor(mu_n,nNodes,1),torch.cmul(W,MA))
		gradXiA = -torch.cmul((b+torch.sum(torch.cdiv(xiprodA,MAA),2)),phiA);

		xiprodB = torch.cmul(torch.repeatTensor(mu_n,nNodes,1),(torch.cmul(W,MB)))
		gradXiB = -torch.cmul(b+torch.sum(torch.cdiv(xiprodB,MBB),2):t(),phiB);
		
		
		gradXi = torch.cdiv(gradXiA + gradXiB, phiA + phiB) + b + torch.mm(W,mu_n:t()):t();
		
		-- update xi
		xi_n = xi_n - torch.mul(gradXi,params.stepSizeXi);
		-- HOW TO CONSTRAIN THE XI TO BE IN (0,1) - just truncate in this region??
		comp1 = xi_n:gt(torch.Tensor(1,nNodes):fill(1))
		comp0 = xi_n:lt(torch.Tensor(1,nNodes):fill(0))
		xi_n[1][comp1] = 1;
		xi_n[1][comp0] = 0;
	end;
	
	-- Update mu_n	
	mu_n = sigmoid(b + torch.mm(W,mu_n:t()) + torch.mm(W:t(),(mu_n - xi_n):t()) + torch.sum(K,1)) -- eq 24, dim (1 x nNode)
	mu_n[{{1},{1,x_n:size(2)}}] = x_n:clone(); -- data set to coord of mu
	
	return mu_n, xi_n, phi, phiA, phiB
end; -- function