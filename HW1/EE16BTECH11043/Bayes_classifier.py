import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# Import Data
df_train_X = pd.read_csv('X.csv',header=None)
df_Y = pd.read_csv('Y.csv',header=None)
train_X = np.array(df_train_X ,dtype=np.float64)
train_X = train_X.T
print train_X.shape

Y = np.array(df_Y,dtype=np.float64)
print Y.shape

X_test = np.array([[1,1],[1,-1],[-1,1],[-1,-1],[1,2]])

###################################################################################################################################
def naive_bayes(x11,x12,x21,x22,x,y):
	mean11=np.mean(x11)
	var11 =np.var(x11)

	mean12=np.mean(x12)
	var12 =np.var(x12)

	mean21=np.mean(x21)
	var21 =np.var(x21)

	mean22=np.mean(x22)
	var22 =np.var(x22)
	pr1=1.0*x11.shape[0]/(x11.shape[0]+x21.shape[0])
	pr2=1.0*x21.shape[0]/(x11.shape[0]+x21.shape[0])

	class1=(np.exp(-(x-mean11)**2/(2*(var11)))/(np.sqrt(2*np.pi*var11)))
	class1=class1*(np.exp(-(y-mean12)**2/(2*(var12)))/(np.sqrt(2*np.pi*var12)))*pr1

	class2=(np.exp(-(x-mean21)**2/(2*(var21)))/(np.sqrt(2*np.pi*var21)))
	class2=class2*(np.exp(-(y-mean22)**2/(2*(var22)))/(np.sqrt(2*np.pi*var22)))*pr2

	print ('class1:')
	print class1
	print ('class-1:')
	print class2

	if class1>class2:
		return 1

	return -1
####################################################################################################################################

print('X_test:')
# x=float(input())
# y=float(input())

#initialisation
flag1=0
flag2=0

for i in range (0,train_X.shape[0]):
	if Y[i]==1:
		flag1=flag1+1

x11=np.zeros(([flag1,1]))
x12=np.zeros(([flag1,1]))
x21=np.zeros(([train_X.shape[0]-flag1,1]))
x22=np.zeros(([train_X.shape[0]-flag1,1]))

#initialisation
flag1=0
flag2=0

for i in range (0,train_X.shape[0]):
	if Y[i]==1:
		x11[flag1]=train_X[i][0]
		x12[flag1]=train_X[i][1]
		flag1=flag1+1
	if Y[i]==-1:
		x21[flag2]=train_X[i][0]
		x22[flag2]=train_X[i][1]
		flag2=flag2+1

for i in range(0,5):
	print X_test[i]
	predict=naive_bayes(x11,x12,x21,x22,X_test[i][0],X_test[i][1])
	print ('The class it belongs to is:')
	print predict
	print('\n')

############################################################################