-- General purpose slice sampler (axis-aligned sampler)
-- The sampler returns all samples generated, thus burnin samples must be removed from the calling function. 

-- Shakir, May 2013.
-- Following MacKay, Chp 29.7, pg 375
-- See also Neal (2005?)

function slicesample(nSamples, logprob, initVec, width, stepOut, params)
	
	
	
	local D = initVec:size(2); -- initVec is 1XD vec
	local samples = torch.Tensor(nSamples, D): zero();
	local x = initVec;
	local maxLoop = 200;
	-- make change later to allow widths for each dimension to be specified as a vector.
	local w = torch.Tensor(1,D):fill(width);
	local logp = logprob(x, params);
	local xLeft, xRight, xprime;
	
	for iter = 1,nSamples do
		print(string.format("Iter %d, logp = %f",iter, logp[1][1]))
		loguprime = torch.log(torch.rand(1)) + logp; -- Inverse CDF sampling for Unif. 
		-- Sample all coordinates
		for d = 1,D do
			xLeft = x:clone(); 
			xRight = x:clone();
			xprime = x:clone();
			
			-- Construct horizontal interval enclosing X
			scale = torch.rand(1,1);
			xLeft[1][d] = x[1][d] - scale[1][1]*w[1][d];
			xRight[1][d] = x[1][d] + (1-scale[1][1])*w[1][d];
			
			-- Stepping out, steps 3(a)-(e)
			local nn = 0;
			if 1== stepOut then		
				local test = torch.squeeze(logprob(xLeft, params)) > torch.squeeze(loguprime);
				while test do
					xLeft[1][d] = xLeft[1][d]  - w[1][d];
					test = torch.squeeze(logprob(xLeft, params)) > torch.squeeze(loguprime);
				end;
					
				test = torch.squeeze(logprob(xRight, params)) > torch.squeeze(loguprime);
				while test do
					xRight[1][d] = xRight[1][d]  + w[1][d];
					test = torch.squeeze(logprob(xRight, params)) > torch.squeeze(loguprime);
				end;		
			end;
					
			local done = false;
			local nLoop = 0;
			while not done do
				nLoop = nLoop + 1;
				xprime[1][d] = torch.rand(1)[1]*(xRight[1][d] - xLeft[1][d]) + xLeft[1][d];
				logp = logprob(xprime, params);
				
				test = torch.squeeze(logp) > torch.squeeze(loguprime);
				if test then
					done = true;
				else
					-- Shrink the interval, steps 8(a),(b)
					test = xprime[1][d] > x[1][d];
					if test then
						xRight[1][d] = xprime[1][d]
					else
						xLeft[1][d] = xprime[1][d]
					end;
				end;
				
				if nLoop >= maxLoop then
					done = true;
				end;
				
				x[1][d] = xprime[1][d]
			end;
		end;
		samples[{{iter},{}}] = x:clone();
	end;
	
	return samples
end;