require("nn")
require("cutorch")
require("cunn")
ffi = require("ffi")
require("parse_data")
require "data"
require "gnuplot"
dofile("config.lua")

function main()
--layer_id =21
--print(table.getn(arg))
if table.getn(arg) < 3 then
	error('The input format should be:\n top_grad_words.lua DATA_FILE FEATURE_FILE SAVE_NAME')
end
print(arg[1])
print(arg[2])
obj = torch.load(arg[2])
input_data = Data(config.val_data)
source = torch.load(arg[1])
save_name = arg[3]
input,labels,n = parse_data(source)
--save_name = arg[3]


sp = n
f = obj.features:double()
l = obj.labels:double()
for i=1,sp do
	if labels[i]~=l[i] then
		error('label dismatch!')
	end
end
input_l = f:size(2)
f=f[{{1,sp},{}}]
print(f:size())
print(input_l)
local i,j
--str_g = torch.Tensor(input_l)
top_grad  ={}
top_n=15
for i = 1,sp do
	--print(i)
	str_g = torch.reshape(f[{{i},{}}],input_l)
	--print(ind:size())
	words = input_data:stringToWords(input[i],input_l)
	--print(words[1][1])
	--print(words[1][2])
	--print(#words)
	words_grad_s = torch.Tensor(#words)
	words_grad_s:zero()

	
	for j=1,#words do
		words_grad_s[j] = torch.sum(str_g[{{words[j][1],words[j][2]}}])
	end
	sorted, idx = torch.sort(words_grad_s,1, true)
	-- nums =linspace(1,#words)
	-- gnuplot.figure(i.."sorted grad sum for words")
	-- gnuplot.plot({"value",nums,sorted})
	-- os.execute("sleep " .. tonumber(5))
	top_grad[i] ={}
	for j=1,math.min(top_n,#words) do
		--%%%%%%reverse order%%%%%%%%%%%
		word_head = #input[i]+1-words[idx[j]][2]
		word_tail = #input[i]+1-words[idx[j]][1]
		--print(word_head)
		--print(word_tail)
		--print(input[i]:sub(word_head,word_tail))
		top_grad[i][j] = input[i]:sub(word_head,word_tail)
	end 

end
local file = io.open(save_name,'w')
for i=1,sp do
	
	file:write(i.." "..labels[i]..' ')
	for j=1,#top_grad[i]-1 do
		file:write(top_grad[i][j]..' ')
	end
	file:write(top_grad[i][#top_grad[i]]..'\n')

end

file:close()
end
main()