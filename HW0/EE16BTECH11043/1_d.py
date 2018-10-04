import numpy as np
import matplotlib.pyplot as plt

# Number of training samples
train_N=500


# Generate equispaced floats in the interval [0, 2pi]
train_x=np.linspace(0,2*np.pi,train_N)

# Generate noise
mean=0
std= 0.05

# Generate some numbers from the sine function
train_y = np.sin(train_x)
train_x=train_x.reshape([train_N,1])

# Add noise
train_y += np.random.normal(mean, std, train_N)
train_y=train_y.reshape([train_N,1])


print ('train_x.shape:')
print train_x.shape
print ('train_x.shape:')
print train_y.shape

###############################################################################################
#Maximum likelihood weight estimation

	#w(optimal)=inv(X^T*X)*X^T*Y
	#let w_opt be the optimum w
	#let test_y be the labels found from the optimal w(Z)
##Graphs plotted for testing data
###############################################################################################

def error(y,y_hat):
	error=np.matmul(np.transpose(y-y_hat),(y-y_hat))
	return error


print('\n______________________________________')
print('\nMaximum likelihood weight estimation')
print('\n______________________________________\n')

#training
ones_arr=np.ones([train_N,1])
train_X=np.array([train_N,2])
train_X=np.append(ones_arr,train_x,axis=1)
w_opt=np.matmul(np.matmul((np.linalg.inv(np.matmul(train_X.T,train_X))),train_X.T),train_y)


#testing
print ('The number of testing samples be:')
test_N=input()

test_x=np.linspace(0,2*np.pi,test_N)+np.random.normal(mean, std, test_N)
test_x=test_x.reshape([test_N,1])
ones_arr=np.ones([test_N,1])
test_X=np.array([test_N,2])
test_X=np.append(ones_arr,test_x,axis=1)
test_y=np.matmul(test_X,w_opt)

print ('testing error:')#here taken cost function to SSE
print error(np.sin(test_x),test_y)

plt.plot(test_x,test_y,'*')
plt.plot(test_x,np.sin(test_x),'r')
plt.show()


# Most likelihood case variance 1/{beta}
print("Give a variance for generating labels :")
std1 = input()
std1 = float(std1)
test_y=test_y.reshape([test_N,])
labels = np.random.normal(test_y,std1,test_N)
variance = error(test_y , labels)/test_N#this is the variance
print("Variance of the error in estimation =",variance)

##################################################################################################
##################################################################################################