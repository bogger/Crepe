require("nn")
require("cutorch")
require("cunn")
require("gnuplot")

-- Local requires
require("data")
require("model")
require("extract")
require 'hdf5'
in_path = '../train_yelp/3m_polarity/train_score_3month_2.t7b'
out_path = '../train_yelp/3m_polarity/train_score_3month_2.h5'
data = torch.load(in_path)
-- print(data)

--feats = torch.totable(data['features']:float())
--labels = torch.totable(data['labels'])
-- print(feats)
--string = json.encode({scores=feats})
-- print(string)
file = hdf5.open(out_path,'w')
file:write('features', data['features']:float())
file:write('labels', data['labels']:int())
file:close()
collectgarbage()
