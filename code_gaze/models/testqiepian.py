import numpy as np

arr=(np.arange(36)).reshape(6,6)
x=np.array([0,1,2,1,4,5])
print(arr[x==1,2:])
