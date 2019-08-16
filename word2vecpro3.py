#-*- coding: UTF-8 -*-


import word2vec
#from gensim.models.word2vec import Word2Vec
from numpyprocess import *

set_printoptions(threshold=NaN)



#with open('corpusSegDone.txt', 'r') as fW:
    #segsentences = fW.readlines()
    #segsentences = (''.join(segsentences))
    #print(segsentences)
word2vec.word2vec('../textprocess1/corpusSegDone.txt', 'corpusWord2Vec.bin', size=6, verbose=True)
model = word2vec.load('corpusWord2Vec.bin')
wordvectors = model.vectors
print (wordvectors)
#vectorstr = (''.join(wordvectors))
#with open('vectors.txt','w') as fW:

     #fW.write(str(wordvectors).encode('utf-8'))


print (model.vocab[0])

