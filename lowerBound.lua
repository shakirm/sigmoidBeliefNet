function lowerBound(mu_n, xi_n, phiA, phiB, W, b, D)
	
	local nNodes = mu_n:size(2);
	
	_mu = mu_n[{{},{D+1,nNodes}}]			
	
	-- hack: to make sure there is no log(0)
	local idx = _mu:eq(0) 
	_mu[idx] = 1e-6;
	negMu = torch.Tensor(_mu:size()):fill(1) - _mu;
	idx = negMu:eq(0);
	negMu[idx] = 1e-6;
	
	-- compute bound 
	KL = torch.sum(torch.cmul(_mu,torch.log(_mu)) + torch.cmul(negMu,torch.log(negMu)))  -- KL term
	expect1 = -torch.sum(torch.log(phiA + phiB));
	expect2 = torch.squeeze(torch.mm((mu_n - xi_n),torch.mm(W,mu_n:t())+ b));
	
	val = (expect1 + expect2 + KL); -- not neg!
	
	
	-- for debug
	if isnan(val) then
		print(expect1)
		print(expect2)
		print(KL)
		print(_mu)
		print(negMu)
		os.exit()
	end;
	
	return val;
end;
