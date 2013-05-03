-- Generate or load binary data sets for testing SBNs
-- Shakir, May 2013

function getData(fname, options, plot)
	local x,y;
	
	if 'bars' == fname then
		print('bars data');
		local nObs = options[1]; -- generate 100 observations
		local imSize =options[2]; -- use 3x3 images
		
		x = torch.Tensor(nObs, imSize*imSize);
		local tmp = torch.Tensor(imSize,imSize):zero();
		for i = 1,nObs do
			tmp:fill(0);
			vert = torch.squeeze(torch.rand(1))>0.5
			if vert then
				
			else
			end;
			
		end;
		
	elseif 'blockImages' == fname then
		print('block Images');
		
	elseif 'mnist' == fname then
		print(mnist);
		
	else
		error('Unknown data set',fname)
	end;
	
	return x,y;
end;

