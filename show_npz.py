import numpy as np
np.set_printoptions(threshold=np.inf)

npzfile=np.load("npz_path")
x=npzfile["data"]
y=npzfile["labels"]
#y=npzfile["labels"]
npzfile.close()
doc=open("txtfile_path",'w')

for i in range(0, len(x)): 
    print(x[i],file=doc)
# for i in x:
#     print(i,file=doc)
doc.close()

#print(x)