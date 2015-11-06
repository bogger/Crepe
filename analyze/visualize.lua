m = require 'manifold'
gfx = require("gfx.js")
require("nn")
require("cutorch")
require("cunn")
ffi = require("ffi")

function main()
layer_id =21
obj = torch.load('../train_yelp/extracted_layer_'..layer_id..'.t7b')
source  = torch.load('../data/yelp_part_test.t7b')




--print(source)
--print(source.)
--text = source[{:,2}]
--rating = source[{}]
sp=100
cls_n=5
f = obj.features:double()
f = f[{{1,sp},{}}]
l = obj.labels:double()
l = l[{{1,sp}}]
print(f:size())
print(l:size())
inputs,labels,n = parse_data(source,sp)
-- check consistency
if n~=sp then
	error('returned length is '..n..' but not '..sp)
end
local i
for i=1,sp do
	if l[i]~=labels[i] then
		error('labels are not match at '..i)
		break
	end
end
print(inputs[1])
--os.exit()
p = m.embedding.tsne(f,{dim=2, perplexity=30})
class_points={}
for i=1,cls_n do
	class_points[i] = {}
	class_points[i].values = {}
	class_points[i].key =  "Class"..i
end
for i=1,sp do
	table.insert(class_points[l[i]].values,
     {
         x = p[i][1],
         y = p[i][2],
         size = 2,
         tag = inputs[i]
	})
end
local config = {
	chart='scatter',
	width = 700,
	height = 700
}
gfx.chart(class_points,config)

end

function parse_data(data,samples)
	local i=1
	local j=0
     if data.index[i] == nil then return end

      local inputs = {}
      local labels = torch.Tensor(samples)

      local n = 0
      for k = 1, samples do
	 j = j + 1
	 if j > data.index[i]:size(1) then
	    i = i + 1
	    if data.index[i] == nil then
	       break
	    end
	    j = 1
	 end
	 n = n + 1
	 local s = ffi.string(torch.data(data.content:narrow(1, data.index[i][j][data.index[i][j]:size(1)], 1)))
	 for l = data.index[i][j]:size(1) - 1, 1, -1 do
	    s = s.." "..ffi.string(torch.data(data.content:narrow(1, data.index[i][j][l], 1)))
	 end
	 inputs[k] = s
	 labels[k] = i
      end

      return inputs, labels,n
end

main()