import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import json
import sys
import pandas
csvpath = '../data/test_review_3month_2_meta.csv'
fig_path = 'test_review_3month_2.png'
fig_title = 'ROC curve from ratings with 3-month prediction'
test_data = pandas.read_csv(csvpath)
labels = test_data['DV'].tolist()
labels = np.array([x-1 for x in labels])
stars = np.array(test_data['stars_y'].tolist())
#labels = labels[stars!=3]
#stars = stars[stars!=3]
scores = [ 1-float(t)/5 for t in stars]
#exit()
class0_n = np.sum(labels==0)
class1_n = np.sum(labels==1)
#true_neg_n = np.sum(scores[labels==0,0] > 0.5)
#true_pos_n = np.sum(scores[labels==1,1] > 0.5)
tot_n = len(labels)
#ac = float(true_neg_n+true_pos_n)/tot_n
#true_neg_r = float(true_neg_n) / class0_n
#true_pos_r = float(true_pos_n) / class1_n

#print 'accuracy: %f' % ac
#print 'true negative rate: %f true postive rate: %f' %(true_neg_r, true_pos_r)
fpr, tpr,_ = roc_curve(labels, scores)
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
