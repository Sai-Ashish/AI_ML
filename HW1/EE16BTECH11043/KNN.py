import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt

#find euclidean distance
def find_euclidean_distance(X,x,y):
	diff_x=X[0]-x
	diff_y=X[1]-y	
	result=np.sqrt(diff_x**2+diff_y**2)
	return result

#KNN function
def KNN(final,k):#k=15 
	class1=0#1
	class2=0#-1
	for i in range (0,k):
		if final[i][1]==1:
			class1=class1+1
		if final[i][1]==-1:
			class2=class2+1
			
	# print (class1-class2)*1.0/k
	if class1>class2 :
		return 1
	else :
		return -1

#sorting
def heapsort(result,n):

	for i in range(n/2-1,-1,-1):
		heapify(result,n,i)

	for i in range(n-1,-1,-1):
		temp=result[0][0]
		result[0][0]=result[i][0]
		result[i][0]=temp
		temp=result[0][1]
		result[0][1]=result[i][1]
		result[i][1]=temp		
		heapify(result,i,0)

def heapify(result,n,i):

	left =2*i+1
	right=2*i+2
	largest=i
	if left<n and result[i][0]<result[left][0]:
		largest=left

	if right<n and result[largest][0]<result[right][0]:
		largest=right

	if largest!=i:

		temp=result[largest][0]
		result[largest][0]=result[i][0]
		result[i][0]=temp

		temp=result[largest][1]
		result[largest][1]=result[i][1]
		result[i][1]=temp			

		heapify(result,n,largest)


#import data
df_X = pd.read_csv('X.csv',header=None)
df_Y = pd.read_csv('Y.csv',header=None)
train_X = np.array(df_X,dtype=np.float64)
train_X = train_X.T
print train_X.shape

Y = np.array(df_Y,dtype=np.float64)
print Y.shape

############################################################################

result=np.zeros(([train_X.shape[0],2]))

#test data
X_test = np.array([[1,1],[1,-1],[-1,1],[-1,-1],[1e-4,0.01]])

#input from user k value
print('k:')
k=int(input())

print('X_test:')

for s in range(0,X_test.shape[0]):
	print X_test[s]
	for i in range (0,train_X.shape[0]):
		result[i][0]=find_euclidean_distance(train_X[i],X_test[s][0],X_test[s][1])
		result[i][1]=Y[i]
	heapsort(result,train_X.shape[0])

	predict=KNN(result,k)

	print ('The class it belongs to is:')
	print predict
	print('\n')

############################################################################