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