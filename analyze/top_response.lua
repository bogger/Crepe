
require("nn")
require("cutorch")
require("cunn")
ffi = require("ffi")
require("parse_data")
function main()
--layer_id =21
--print(table.getn(arg))
if table.getn(arg) < 3 then
	error('The input format should be:\n top_response.lua DATAFILE FEATURE_FILE SAVE_FILE')
end
print(arg[1])
print(arg[2])
obj = torch.load(arg[2])
source  = torch.load(arg[1])
save_name = arg[3]

inputs,labels,n = parse_data(source)
sp = n

--print(source)
--print(source.)
--text = source[{:,2}]
--rating = source[{}]
--sp=source:size(1)
cls_n=2
f = obj.features:double()
print(f:size())
--f = f[{{1,sp},{}}]
if f:dim()<=2 then
	error('f dim should be 3!')
end
spot = torch.reshape(f[{{1,sp},{1},{}}],sp,f:size(3))
filters = f:size(3)

l = obj.labels:double()
l = l[{{1,sp}}]


-- check consistency

local i
local j
for i=1,sp do
	if l[i]~=labels[i] then
		error('labels are not match at '..i)
		break
	end
end
top_n = 10
top_res = {}
top_v = {}
for i=1,filters do
	res = spot[{{},{i}}]
	sorted, indices = torch.sort(res, 1, true)
	print(sorted[1][1])
	top_res[i] = {}
	top_v[i] = sorted[1][1]
	for j=1,top_n do 
		top_res[i][j] = inputs[indices[j][1]]
	end
end

-- file = torch.DiskFile(save_name, 'w')
-- file:writeObject(top_res)
-- file:close()
local file = io.open(save_name,'w')
for i=1,filters do
	if top_v[i]>0 then
		file:write(i.." ")
		for j=1,top_n-1 do
			file:write(top_res[i][j])
			file:write(" ")
		end
		file:write(top_res[i][top_n])
		file:write("\n")
	end
end
file:close()
end

main()