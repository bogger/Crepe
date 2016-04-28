import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import json
import sys
def softmax(x, axis=1):
    """Compute softmax values for each sets of scores in x."""
    sf = np.exp(x)
    sf = sf/np.sum(sf, axis=axis)[:,np.newaxis]
    return sf
json_path = sys.argv[1]#'../train_yelp/3m/extracted_layer_23.json'
feat = json.load(open(json_path))
print type(feat)
#fig_path = sys.argv[2]
scores = feat['scores']
labels = feat['labels']

print type(scores)
scores = np.array(scores)
scores = softmax(scores)
score_sums = np.sum(scores,axis=1)
print score_sums[:10]
pos_scores = scores[:,1]
labels = np.array(labels) - 1#1-base to 0-base
class0_n = np.sum(labels==0)
class1_n = np.sum(labels==1)*20
true_neg_n = np.sum(scores[labels==0,0] > 0.5)
true_pos_n = np.sum(scores[labels==1,1] > 0.5)*20
tot_n = class0_n + class1_n
ac = float(true_neg_n+true_pos_n)/tot_n
true_neg_r = float(true_neg_n) / class0_n
true_pos_r = float(true_pos_n) / class1_n

print 'accuracy: %f' % ac
print 'true negative rate: %f true postive rate: %f' %(true_neg_r, true_pos_r)

