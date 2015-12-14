import numpy as np
import scipy.io as sio
from time import time
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter
from stop_words import get_stop_words
def print_top_words(save_name,model, feature_names, n_top_words):
	with open(save_name,'w') as outfile:
	    for topic_idx, topic in enumerate(model.components_):
	        outfile.write("Topic #%d:\n" % topic_idx)
	        outfile.write(" ".join([feature_names[i]
	                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
	        outfile.write("\n")
    #print()

dataFile='../train_yelp_polarity/top_grad_word.txt'
data=[]
all_words=[]
target_label= '2'
with open(dataFile,'r') as infile:
	for line in infile:
		items = line.split()
		if items[1]==target_label:
			data.append(items[2:])
			all_words.extend(items[2:])
print data[0]
print all_words[0]
#transform word list to word matrix
top_word_n = 1000
top_topic_words = 5
stop_words = get_stop_words('en')
c= Counter(all_words).most_common(top_word_n)#generate a dict
#filter the top_words
c = [w for w in c if w[0] not in stop_words]


print len(c)
top_word_n = len(c)
w_times = np.asarray([w[1] for w in c],dtype=float)
total_times = np.sum(w_times)
word_list = {}
words=[]

for i in xrange(top_word_n):
	word_list[c[i][0]] = i
	words.append(c[i][0])

with open('lda_words.txt','w') as fout:
	for w in words:
		fout.write(w+' ')
#exit()
print c[0][0]
data_n = len(data)
word_matrix = np.zeros((data_n,top_word_n),dtype=np.int32)
for idx,review in enumerate(data):
	for w in review:
		if w in word_list:
			word_matrix[idx][word_list[w]] = word_matrix[idx][word_list[w]]+1#1.0/c[word_list[w]][1]#tfidf
n_topics = 10
model = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                learning_method='online', 
                                random_state=0)

print 'start to train LDA'
t0 = time()
model.fit(word_matrix)
p_t = model.transform(word_matrix)#p_t is unnormalized!
p_t = p_t / np.tile(p_t.sum(1).reshape(data_n,1),[1,n_topics])
print p_t[0],p_t[1]
p_t_w = model.components_
print p_t_w.shape
#entropy of topics
p_t_all=np.reshape(np.mean(p_t,0),[n_topics,1])
print p_t_all.shape
e_t = - p_t_all * np.log(p_t_all) - (1.0 - p_t_all) * np.log(1-p_t_all)
p_w = np.reshape(w_times / total_times,[1,top_word_n])
print p_w.shape
print np.tile(p_t_all, [1,top_word_n]).shape
print np.tile(p_w,[n_topics,1]).shape
#p_w_t = p_t_w * np.tile(p_t_all, [1,top_word_n]) / np.tile(p_w,[n_topics,1])
e_w_t = np.zeros((n_topics, top_word_n),dtype=float)
p_w_t = np.zeros((n_topics, top_word_n),dtype=float)
p_nw_t = np.zeros((n_topics, top_word_n),dtype=float)
p_w = np.zeros((1,top_word_n),dtype=float)
p_w = np.mean(word_matrix>0,0)
print p_w
print p_t[0].shape
for j in xrange(top_word_n):
	temp = np.asarray([p_t[i] for i in xrange(data_n) if word_matrix[i][j]>0])
	print temp.shape
	p_w_t[:,j] = np.mean(temp,0)
	temp = np.asarray([p_t[i] for i in xrange(data_n) if word_matrix[i][j]==0])
	p_nw_t[:,j] = np.mean(temp,0)
e_w_t = np.tile(p_w,[n_topics,1]) *(- p_w_t * np.log(p_w_t) - (1-p_w_t) * np.log(1-p_w_t) ) \
        +np.tile(1-p_w,[n_topics,1]) * (- p_nw_t * np.log(p_nw_t) - (1-p_nw_t) * np.log(1-p_nw_t))
#mutual information (inversed)
mi = -np.tile(e_t,[1,top_word_n]) + e_w_t
top_words_idx = np.argsort(mi,1)[:,0:top_topic_words]
print [mi[0][i] for i in top_words_idx[0]]
print 'done in %.3f seconds.' % (time()-t0)
print('Print top topic words')
#print_top_words('top_topic_words_pos.txt',model, words, top_topic_words)
outfile='top_mi_words_pos.txt'
with open(outfile,'w') as out:
	for i in xrange(n_topics):
		out.write('Topic #%d\n' % i)
		out.write(' '.join([words[k] for k in top_words_idx[i]]))
		out.write('\n')
