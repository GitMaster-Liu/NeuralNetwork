import numpy as np

f = open(r"textdigit2.data")
line = f.readline()
data_list = []
while line:
    num = list(map(float,line.split()))
    data_list.append(num)
    line = f.readline()
f.close()
data_array = np.array(data_list)
print data_array
print type(data_array)
print data_array[1]
