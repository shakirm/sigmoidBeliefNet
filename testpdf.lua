-- test function for slice sampling

-- A multimodal function
function testpdf(x,params)
	
	local term1 = torch.exp(-torch.mul(torch.pow(x,2),0.5));
	local term2 = torch.add(torch.pow(torch.sin(torch.mul(x,3)),2),1);
	local term3 = torch.add(torch.pow(torch.cos(torch.mul(x,5)),2),1);
	
	local logp = torch.log(torch.cmul(torch.cmul(term1, term2),term3));	
	return logp;
end;

-- 2D Gaussian with mean and Covariance
function bivariateGauss(x,params)
	local m = params.mu;
	local S = params.cov;
	
	local diff = torch.add(x,-m);
	local Sinv = torch.inverse(S)
		
	local logp = -torch.mul(torch.mm(torch.mm(diff,Sinv),diff:t()),0.5);
	return logp;
end;