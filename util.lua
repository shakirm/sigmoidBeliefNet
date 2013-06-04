-- Commonly used utility functions
-- SHakir, April 2013

-- Compute sigmoid function: 1/1+exp(-x)
function sigmoid(x)
        
        local eta = torch.exp(-x);
        local val = torch.pow(eta:add(1),-1); 
        
        return val;
end;

function isnan(x)
    return not (x == x);
end;

function randperm(n)
    -- returns a random permutation of integers 1 to n.
    -- note: changes state of seed through call to rand.
    
    tmp, idx = torch.sort(torch.rand(n));
    return idx;
end;