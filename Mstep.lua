

function Mstep(W, b, gradW, gradB, stepSize, weightMask)
	
	W = W - torch.mul(gradW, stepSize);
	W = torch.cmul(W,weightMask);
	b = b - torch.mul(gradB,stepSize);
	
	return W, b;
end; --function

