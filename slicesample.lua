-- General purpose slice sampler 
-- The sampler returns all samples generated, thus burnin samples must be removed from the calling function. 

-- Shakir, May 2013.
-- Following MacKay, Chp 29.7, pg 375

function slicesample(nSamples, logprob, initVec, width, stepOut, params)
	
	local D = initVec:size(2); -- initVec is 1XD vec
	local samples = torch.Tensor(nSamples, D): zero();
	local x = initVec;
	local maxLoop = 200;
	-- make change later to allow widths for each dimension to be specified as a vector.
	local w = torch.Tensor(1,D):fill(width);
	print(w)
	local logp = logprob(x, params);
	print(logp)
	
	local xLeft, xRight, xprime;
	
	for iter = 1,nSamples do
		
		loguprime = torch.log(torch.rand(1)) + logp; -- Inverse CDF sampling for Unif. 
		print(loguprime)
		-- Sample all coordinates
		for d = 1,D do
			xLeft = x; 
			xRight = x;
			xprime = x;
			
			-- Construct horizontal interval enclosing X
			offset = torch.rand(1,1);
			xLeft[{{1},{d}}] = xLeft[{{1},{d}}] - offset*torch.squeeze(w[{{1},{d}}]);
			print(xLeft)
			
			xRight[{{1},{d}}] = xRight[{{1},{d}}] + torch.add(offset,-1)*torch.squeeze(w[{{1},{d}}]);
			
			print(xRight)
		
			
			-- Stepping out, steps 3(a) - (e)
			if 1== stepOut then
				print(tensor.squeeze(logprob(xLeft, params)))
				while tensor.squeeze(logprob(xLeft, params)) > torch.squeeze(loguprime) do
					xLeft[{{1},{d}}] = torch.squeeze(xLeft[{{1},{d}}])  - torch.squeeze(w[{{1},{d}}]);
				end;
				
				while logprob(xRight, params) > loguprime do
					xRight[{{1},{d}}] = torch.squeeze(xRight[{{1},{d}}])  + torch.squeeze(w[{{1},{d}}]);
				end;		
			end;
			
			local done = false;
			local nLoop = 0;
			while not done do
				nLoop = nLoop + 1;
				
				xprime[{{1},{d}}] = torch.rand(1)*(xRight[{{1},{d}}] - xLeft[{{1},{d}}]) + xLeft[{{1},{d}}];
				logp = logprob(xprime, params);
				
				if logp > loguprime then
					done = true;
				else
					-- Shrink the interval, steps 8(a),(b)
					if xprime[{{1},{d}}] > x[{{1},{d}}] then
						xRight[{{1},{d}}] = xprime[{{1},{d}}]
					else
						xLeft[{{1},{d}}] = xprime[{{1},{d}}]
					end;
				end;
				
				if nLoop >= maxLoop then
					done = true;
				end;
				
				x[{{1},{d}}] = xprime[{{1},{d}}]
			end;
		end;
		
		samples[{{iter},{}}] = x;
	end;
	
	return samples
end;