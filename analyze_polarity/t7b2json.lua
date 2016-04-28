require("nn")
require("cutorch")
require("cunn")
require("gnuplot")

-- Local requires
require("data")
require("model")
require("extract")
json=require("json")
in_path = '../yelp_polarity/train_score_3month_2.t7b'
out_path = '../yelp_polarity/train_score_3month_2.json'
data = torch.load(in_path)
-- print(data)

feats = torch.totable(data['features']:float())
--labels = torch.totable(data['labels'])
-- print(feats)
string = json.encode({scores=feats})
-- print(string)
json_file = io.open(out_path,'w')
json_file:write(string)
json_file:close()
collectgarbage()
