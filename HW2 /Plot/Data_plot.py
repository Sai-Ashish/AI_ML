import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt

#import data
df_X = pd.read_csv('Xsvm.csv',header=None)
df_Y = pd.read_csv('ysvm.csv',header=None)
train_X = np.array(df_X,dtype=np.float64)

Y = np.array(df_Y,dtype=np.float64)
flag1=0

for i in range (0,train_X.shape[0]):
	if Y[i]==1:
		flag1=flag1+1

class1=np.zeros(([flag1,2]))
class2=np.zeros(([500-flag1,2]))
k=0
l=0
for i in range(0,train_X.shape[0]):
	if Y[i]==1:
		class1[k][0]=train_X[i][0]
		class1[k][1]=train_X[i][1]
		k=k+1
	else:
		class2[l][0]=train_X[i][0]
		class2[l][1]=train_X[i][1]
		l=l+1

plt.plot(class1[:,0],class1[:,1],'o',class2[:,0],class2[:,1],'ro')
plt.legend(['y = 1', 'y = -1'], loc='best')
plt.show()