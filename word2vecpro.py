# -*- coding: utf-8 -*-

from gensim.models.word2vec import Word2Vec
import os



filePath="corpusSegDone.txt"
#with open (filePath, 'r', encoding='utf-8') as f:
    #sentences=f.readlines()
#print(sentences)
sentences=[['CC','>','CCHAP','β','CV','≥','CFHHR'],['CC','>','CCHAP','β','CV','<','CFHHR'],['CH','<','CLHAP','β','CV','≥','CFHHR'],['CH','<','CLHAP','β','CV','<','CFHHR'],['BC','>','BCHAP','γ','BV','≥','BFHHR'],['BC','>','BCHAP','γ','BV','<','BFHHR'],['BH','<','CLHAP','γ','BV','≥','BFHHR'],['BH','<','CLHAP','γ','CV','<','BFHHR'],['LH','>','LCHAP'],['LH','<','LCLAP'],['LH','<','LCLSP'],['LH','>','LCHSP']]
sentences=str(sentences)
sentencespro=sentences.replace('>','大于')
print (sentences)
if os.path.exists("MyModel"):
    model = Word2Vec.load('MyModel')
else:
    model = Word2Vec(sentences, size=100, min_count=5, workers=2, iter=50)
    model.save('MyModel')
print(model.most_similar('CV'))
print(model['CV'])
#wordvectors=(model[u'β'])
#print(wordvectors)
#with open('corpusWord2Vec.txt','wb') as fW:
    #fW.write((wordvectors))
    #fW.write('\n'.encode('utf-8'))