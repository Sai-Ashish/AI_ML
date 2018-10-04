import numpy as np
import matplotlib.pyplot as plt

# Number of training samples
train_N=10


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

################################################################################################
#Vanilla linear regression

	#w(optimal)=inv(X^T*X)*X^T*Y
	#let w_opt be the optimum w
	#let y_pred (train),test_y (test) be the labels found from the optimal w(Z)

################################################################################################

def error(y,y_hat):
	error=np.matmul(np.transpose(y-y_hat),(y-y_hat))
	return error


print('\n_______________________________')
print('\nVanilla linear regression')
print('\n_______________________________\n')

#training
ones_arr=np.ones([train_N,1])
train_X=np.array([train_N,2])
train_X=np.append(ones_arr,train_x,axis=1)

w_opt=np.matmul(np.matmul((np.linalg.inv(np.matmul(train_X.T,train_X))),train_X.T),train_y)
y_pred = np.matmul(train_X,w_opt)

#trining error
print ('training error:')
print error(train_y,np.matmul(train_X,w_opt))


plt.plot(train_x,train_y,'r',label='True values')
plt.plot(train_x,y_pred,'*',label='estimated')
plt.show()


#testing
print ('\nThe number of testing samples be:')
test_N=int(input())
test_x=np.linspace(0,2*np.pi,test_N)+np.random.normal(mean, std, test_N)
test_x=test_x.reshape([test_N,1])
ones_arr=np.ones([test_N,1])
test_X=np.array([test_N,2])
test_X=np.append(ones_arr,test_x,axis=1)
test_y=np.matmul(test_X,w_opt)

print ('testing error:')
print error(np.sin(test_x),test_y)


plt.plot(test_x,test_y,'*')
plt.plot(test_x,np.sin(test_x),'r')
plt.show()


################################################################################################
################################################################################################