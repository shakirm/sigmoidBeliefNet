-- Generate or load binary data sets for testing SBNs
-- Shakir, May 2013

require 'image'
require 'dataset/cifar10'
require 'dataset/mnist'
require 'dataset/smallnorb'
 
function getData(fname, options, plot)
	local x,y;
	
	if 'bars' == fname then
		print('bars data');
	
		local nObs = options[1]; -- generate 100 observations
		local imSize = torch.sqrt(options[2]); -- use 3x3 images
		
		x = torch.Tensor(nObs, imSize*imSize);
		local tmp = torch.Tensor(imSize,imSize):zero();
		for i = 1,nObs do
		   local d = torch.zeros(imSize,imSize)
		   -- flip one column chosen with proba 1/3
		   local col = torch.random(1,3)
		   d[{{},col}] = 1
		   -- transpose to horizontal with proba 1/3
		   if torch.rand(1)[1] < .3 then d = d:t() end
		   -- flip to white on black with proba 1/2
		   if torch.rand(1)[1] < .5 then
			  d:apply(function(x) return 1 - x end)
		   end
		   
		   --gnuplot.imagesc(d);
		   local tmp = d:resize(1,imSize*imSize);
		   x[{{i},{}}] = tmp:clone();
		end;
		
	elseif 'blockImages' == fname then
		print('block Images');
			
		local nObs = options[1]; -- generate 100 observations
		local imSize =options[2];

		Z = torch.rand(nObs,4):gt(0.66);
		A = torch.Tensor(4,imSize):zero();
		
		f1idx = {2, 7, 8, 9, 14};
		f2idx = {4, 5, 6, 10, 12, 16, 17, 18};
		f3idx = {19, 25, 26, 31, 32, 33};
		f4idx = {22, 23, 24, 29, 35};
		
		for i = 1, #f1idx do A[{{1},{f1idx[i]}}] = 1; end;
		for i = 1, #f2idx do A[{{2},{f2idx[i]}}] = 1; end;
		for i = 1, #f3idx do A[{{3},{f3idx[i]}}] = 1; end;
		for i = 1, #f4idx do A[{{4},{f4idx[i]}}] = 1; end;
		
		Xclean = torch.mm(Z:double(),A);
		
		x = Xclean:clone();
		for i = 1,nObs do
			for j = 1,imSize do
				if torch.rand(1)[1] < 0.05 then
					x[{{i},{j}}] = 1 - torch.squeeze(Xclean[{{i},{j}}]);
				end;
			end;
		end;
		
	elseif 'mnist' == fname then
		print('MNIST');
		local imSize = 28*28;
	
		local tmp = Mnist.dataset({test = false}) -- non-scaled MNIST
        -- testData = Mnist.dataset({test = true})
		-- isRGB = false
	    
	    print(tmp)
	    local nObs = tmp.dataset.data[1]:nElement();
	    local x = torch.Tensor(nObs,imSize)
	    local y = tmp.dataset.class;
	    for i = 1,nObs do
			-- gnuplot.imagesc(tmp.dataset.data[i][1])
	    	x[i] = tmp.dataset.data[i][1]:resize(1,imSize);
	    end;
	    
		-- gnuplot.imagesc(x)
	    -- gnuplot.hist(y)
	    
	else
		error('Unknown data set',fname)
	end;
	
	return x,y;
end;