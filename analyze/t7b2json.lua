require("nn")
require("cutorch")
require("cunn")
require("gnuplot")

-- Local requires
require("data")
require("model")
require("extract")
json=require("json")
in_path = '../train_yelp/test_score_3month_2.t7b'
out_path = '../train_yelp/test_score_3month_2.json'
data = torch.load(in_path)
-- print(data)

feats = torch.totable(data['features'])
labels = torch.totable(data['labels'])
-- print(feats)
string = json.encode({scores=feats,labels=labels})
-- print(string)
json_file = io.open(out_path,'w')
json_file:write(string)
json_file:close()
collectgarbage()
