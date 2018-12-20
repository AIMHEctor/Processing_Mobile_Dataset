import numpy as np
np.set_printoptions(threshold=np.inf)
#npzfile=np.load("E:/DATA/tor_time_test4w_200w_100tr.npz")

npzfile=np.load("C:/Users/Administrator/Desktop/dlwf_test/data.npz")
x=npzfile["data"]
y=npzfile["labels"]
#y=npzfile["labels"]
npzfile.close()
doc=open("C:/Users/Administrator/Desktop/test.txt",'w')

for i in range(0, len(x)): 
    print(x[i],file=doc)
# for i in x:
#     print(i,file=doc)
doc.close()

#print(x)