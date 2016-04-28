import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import json
import sys
import pandas

def softmax(x, axis=1):
    """Compute softmax values for each sets of scores in x."""
    sf = np.exp(x)
    sf = sf/np.sum(sf, axis=axis)[:,np.newaxis]
    return sf
json_path = '../yelp_polarity/test_score_3month_2.json'
csv_path = '../data/test_review_3month_2_meta.csv'
fig_title = 'ROC curve with combined score of polarity prediction and user ratings'
test_data = pandas.read_csv(csv_path)
labels_rating = test_data['DV'].tolist()
labels_rating = np.array(labels_rating) - 1
print labels_rating.shape
stars = np.array(test_data['stars_y'].tolist()).astype('float32')
scores_rating = stars/5
print max(scores_rating)
print min(scores_rating)
rating_n = len(labels_rating)
feat = json.load(open(json_path))
index = labels_rating.argsort(kind='mergesort')
print index.dtype
#index = index.transpose()
#index = index.astype('int32')
labels_rating = labels_rating[index]
scores_rating = scores_rating[index]
print type(feat)
fig_path = 'test_review_3month_combine.png'
scores = feat['scores']
labels = feat['labels']
print rating_n

print type(scores)
scores = np.array(scores)
scores = scores[:rating_n,:]
scores = softmax(scores)
score_sums = np.sum(scores,axis=1)
print score_sums[:10]
neg_scores = scores[:,0]

labels = np.array(labels) - 1#1-base to 0-base
labels = labels[:rating_n]
tot_n = len(labels)
print tot_n
assert(tot_n == rating_n)
assert((labels_rating == labels).all())
match_pos_n = np.sum((neg_scores < 0.5) & (scores_rating > 0.5))
print float(match_pos_n)/np.sum(scores_rating > 0.5)
rlist = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1]
for r in rlist:
	combined_scores = r*neg_scores + (1-r)*(1 - scores_rating)
	fpr, tpr,_ = roc_curve(labels, combined_scores)
	roc_auc = auc(fpr, tpr)
	print roc_auc
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve with 3-month prediction boundary')
plt.legend(loc="lower right")
plt.savefig(fig_path)
