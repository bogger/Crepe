import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import json
import h5py
import sys
import pandas
import datetime
def softmax(x, axis=1):
    """Compute softmax values for each sets of scores in x."""
    sf = np.exp(x)
    sf = sf/np.sum(sf, axis=axis)[:,np.newaxis]
    return sf
stage='validate'
fig_path = '%s_review_3month_group_month_%%s.png'% stage
feat_path = '../yelp_polarity/%s_score_3month_2.json' % stage
csv_path = '../data/%s_review_3month_2_meta.csv' % stage
fig_title = 'ROC curve with reviews in one month (with the maximum rating score) - %s data' % stage
test_data = pandas.read_csv(csv_path)
labels = test_data['DV'].tolist()
bids = test_data['business_id'].tolist()
dates = test_data['date'].tolist()

labels = np.array(labels) - 1

stars = np.array(test_data['stars_y'].tolist()).astype('float32')
scores_rating = 1-stars/5
print max(scores_rating)
print min(scores_rating)
rating_n = len(labels)
#h5file = h5py.File(feat_path,'r')
#scores = h5file['features']
json_data = json.load(open(feat_path))
scores = json_data['scores']
#find the index projection from input data to features
index = labels.argsort(kind='mergesort')
print index.dtype
#index = index.transpose()
#index = index.astype('int32')
#reorder to match feature order
labels = labels[index]
scores_rating = scores_rating[index]
bids = [bids[x] for x in index]
dates =[datetime.datetime.strptime(dates[x],'%Y-%m-%d') for x in index]
month_tags = [t.strftime('%Y-%m') for t in dates]
bid_month_tags = [bid + month_tag for bid,month_tag in zip(bids, month_tags)] 
#all_info = zip(id_month_tags,labels_rating, scores_rating)
#print type(feat)


#labels = feat['labels']
print rating_n

print type(scores)
scores = np.array(scores)
scores = scores[:rating_n,:]
scores = softmax(scores)
score_sums = np.sum(scores,axis=1)
print score_sums[:10]
neg_scores = scores[:,0]
data_src=[neg_scores, scores_rating]
policies=['score','rating']
for scores_eval,policy in zip(data_src, policies):
#scores_eval = neg_scores # neg_scores or scores_rating
	all_info = zip(bid_month_tags, labels, scores_eval)
	unique_values = set(bid_month_tags)
	data_group_by_month_idx = {}
	for bid_month_tag, label, score in all_info:
		if bid_month_tag in data_group_by_month_idx:
			data_group_by_month_idx[bid_month_tag].append([label,score])
		else:
			data_group_by_month_idx[bid_month_tag] = [[label,score]]
	data_group_by_month = [np.array(x) for k,x in data_group_by_month_idx.iteritems()]
	n_group_by_month = np.array([x.shape[0] for x in data_group_by_month])
	mean_n_group = n_group_by_month.mean()
	print 'mean length of a group is %d' % mean_n_group
	data_n = len(data_group_by_month)
	print '%d data points after merging' % data_n
	labels_grouped = np.zeros((data_n,1))
	scores_grouped = np.zeros((data_n,1))
	#alpha = 2
	for i, group in enumerate(data_group_by_month):
		labels_g = group[:,0]
		scores_g = group[:,1]
		labels_grouped[i] = labels_g.any()
		scores_grouped[i] = scores_g.min()
	#rlist = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1]
	#for r in rlist:
		#combined_scores = r*neg_scores + (1-r)*(1 - scores_rating)
	fpr, tpr,_ = roc_curve(labels_grouped, scores_grouped)
	roc_auc = auc(fpr, tpr)
	print roc_auc
	plt.figure()
	plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title(fig_title)
	plt.legend(loc="lower right")
	plt.savefig(fig_path % policy)
