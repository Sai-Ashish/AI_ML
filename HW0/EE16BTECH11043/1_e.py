import numpy as np
import matplotlib.pyplot as plt

# Number of training samples
train_N=100

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

##############################################################################################################################
#Linear regression using a polynomial basis function and l2 regularisation

	#zero centered
	#w(optimal)=inv(PHI^T*PHI+l*I)*PHI^T*Y
	#phi used is polynomial basis matrix
	#l is the Lagrangian multiplier
	#let w_opt be the optimum w
	#let test_y be the labels found from the optimal w(w_opt)
	#with increasing training sample incresing order has a good effect
 	#else underfit on test data
#graph plotted for test data 
##############################################################################################################################

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


print('\n_________________________________________')
print ('\nMaximum posterior weight estimation\n')
print('\n_________________________________________')


#training

print("Enter alpha (standard deviation of weights)")
alpha = input()
alpha = float(alpha)

print("Enter sigma (standard deviation of labels)")
sigma = input()
sigma = float(sigma)

l = ((sigma**2)/(alpha**2))
print ('Lagrangian equivalence:') 
print l

print ('The order of the polynomial be:')
order=int(input())+1

train_phi=polynomial_basis(train_x,train_N,order)
train_phi = np.delete(train_phi, 0, axis=1)
# print train_phi
train_y = train_y - np.mean(train_y)

w_opt=np.matmul(np.matmul((np.linalg.inv(np.matmul(train_phi.T,train_phi)+l*np.identity(order-1))),train_phi.T),train_y)

#testing

print ('The number of testing samples be:')
test_N=int(input())

test_x=np.linspace(0,2*np.pi,test_N)+np.random.normal(mean, std, test_N)
test_x=test_x.reshape([test_N,1])

test_phi=polynomial_basis(test_x,test_N,order)
test_phi = np.delete(test_phi, 0, axis=1)

test_y=np.matmul(test_phi,w_opt)+ np.mean(train_y)


print ('testing error:')
print error(np.sin(test_x),test_y)+l*(np.matmul(w_opt.T,w_opt))

plt.plot(test_x,test_y,'*')
plt.plot(test_x,np.sin(test_x),'r')
plt.show()


################################################################################################################################
################################################################################################################################

