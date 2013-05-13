-- My set of utilities to save to a csv file.

function save_csv(fname, matrix)
	local f = io.open(fname,'w')
	if matrix:dim() ~= 2 then error ('2D matrices only') end
	for i=1,matrix:size(1) do
		local str = table.concat(matrix:select(1,i):clone():storage():totable(),',')
		f:write(str .. '\n')
	end
	f:close()
end