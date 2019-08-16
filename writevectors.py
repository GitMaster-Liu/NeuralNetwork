import random


for file_num in range(1,9):
    wordvector = ''
    wordvectors = ''
    for i in range(10):
        p = random.sample(range(21), 1)
        print (p)
        with open('vectors'+'.txt', 'r') as fW:
            data = fW.readlines()
        wordvector=data[p[0]]
        wordvectors=wordvectors+wordvector
    with open('vector' + file_num.__str__() + '.data', 'w') as fm:
        fm.write(wordvectors)