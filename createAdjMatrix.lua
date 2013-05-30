function createAdjMatrix(dims)
	-- create the adjaceny matrix to represent the network structure
	-- Assume that every node is connected to all other nodes in the layer beneath it. This does not include connections between nodes.
	local N = torch.sum(dims);
	local nLayers = (#dims)[1];
	local adjMatrix = torch.Tensor(N,N):zero();
	local meanVec = torch.Tensor(1,N):zero();

	local a0 = 0; 
	local a1 = 0;
	local b0 = 1; 
	local b1 = dims[1];
	
	for i = 1,nLayers-1 do
		tmp = b1
		a0 = b0;
		a1 = tmp;
		b0 = tmp +1;
		b1 = tmp + dims[i+1]

		for p = a0, a1 do
			for q = b0, b1 do
				adjMatrix[p][q] = 1;
			end;
		end;
	end;
	
	meanVec[{{1},{1,N - dims[nLayers]}}] = 1;
	
	return adjMatrix, meanVec;
end;