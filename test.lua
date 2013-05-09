
--function SampleKirby()

	for i = 1, 20
   local d = torch.zeros(3,3)
   -- flip one column chosen with proba 1/3
   local col = torch.random(1,3)
   d[{{},col}] = 1
   -- transpose to horizontal with proba 1/3
   if torch.rand(1)[1] < .3 then d = d:t() end
   -- flip to white on black with proba 1/2
   if torch.rand(1)[1] < .5 then
      d:apply(function(x) return 1 - x end)
   end
   		DD = 
   
   end;
   
   print(d)
   --return d
--end

