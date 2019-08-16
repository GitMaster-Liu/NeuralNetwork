##!/usr/bin/env python
## coding=utf-8
'''use python3.7'''


import jieba

# 数据清洗，判断字符类型
def is_uchar(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar<=u'\u9fa5':
            return True
    """判断一个unicode是否是数字"""
    if uchar >= u'\u0030' and uchar<=u'\u0039':
            return True
    """判断一个unicode是否是英文字母"""
    if (uchar >= u'\u0041' and uchar<=u'\u005a') or (uchar >= u'\u0061' and uchar<=u'\u007a'):
            return True
    if uchar in ('>','<','≥','≤','=','α','β','γ','/','(',')','.'):
            return True
    return False

filePath = 'corpus.txt'
fileSegWordDonePath ='corpusSegDone.txt'
# read the file by line
with open (filePath, 'r', encoding='GBK') as f:
    data = f.readlines()

#data = ''.join(data)
#data = [char for char in data if is_uchar(char)]
#data = ''.join(data)
fileTrainRead = data
#fileTestRead = []

# define this function to print a list with Chinese
def PrintListChinese(list):
    for i in range(len(list)):
        print (list[i],)

# segment word with jieba
#fileTrainRead = [char for char in fileTrainRead if is_uchar(char)]
#fileTrainRead = ''.join(fileTrainRead)
for i in range(len(fileTrainRead)):
    print(fileTrainRead[i])
    fileTrainRead[i]=(('  '.join(jieba.cut(fileTrainRead[i]))))
    print(fileTrainRead[i])

# to test the segment result
#PrintListChinese(fileTrainSeg[10])

# save the result
with open(fileSegWordDonePath,'wb') as fW:
    for i in range(len(fileTrainRead)):
        fW.write(fileTrainRead[i].encode('utf-8'))