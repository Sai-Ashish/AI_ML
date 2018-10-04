import numpy as np
import matplotlib.pyplot as plt

# Number of training samples
train_N=10

# Generate equispaced floats in the interval [0, 2pi]
train_x=np.linspace(0,2*np.pi,train_N)

# Generate noise
mean=0
std= 0.1

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


# ##########################################################################################################
# #Linear regression using a polynomial basis function

# 	#w(optimal)=inv(PHI^T*PHI)*PHI^T*Y
# 	#phi used is polynomial basis matrix
# 	#let w_opt be the optimum w
# 	#let y_pred (train),test_y (test) be the labels found from the optimal w(Z)
# 	#with increasing training sample incresing order has a good effect
#  	#else underfit on test data
# ##########################################################################################################

def error(y,y_hat):
	error=np.matmul(np.transpose(y-y_hat),(y-y_hat))
	return error



def polynomial_basis(x,N,order):
	ones_arr=np.ones([N,1])
	phi=np.append(ones_arr,x,axis=1)
	for i in range(0,order-2):
		phi=np.append(phi,x,axis=1)
	phi=np.power(phi, np.arange(order))
	return phi


print('\n_____________________________________________________')
print('\nLinear regression using a polynomial basis function')
print('\n_____________________________________________________')


#training

print ('The order of the polynomial be:')
order=int(input())+1


train_phi=polynomial_basis(train_x,train_N,order)

w_opt=np.matmul(np.matmul((np.linalg.inv(np.matmul(train_phi.T,train_phi))),train_phi.T),train_y)

y_pred = np.matmul(train_phi,w_opt)



#testing
print ('The number of testing samples be:')
test_N=int(input())

test_x=np.linspace(0,2*np.pi,test_N)+np.random.normal(mean, std, test_N)
test_x=test_x.reshape([test_N,1])
test_phi=polynomial_basis(test_x,test_N,order)
test_y=np.matmul(test_phi,w_opt)

print ('testing error:')
print error(np.sin(test_x),test_y)
std1=.05
plt.plot(test_x,test_y,'*',label='Estimated')
plt.plot(test_x,np.sin(test_x),'r',label='True Values')
test_y=test_y.reshape([test_N,])
labels = np.random.normal(test_y,std1,test_N)
plt.plot(test_x,labels,'.',label='Generated labels from Y_estim')
plt.show()


############################################################################################################
############################################################################################################