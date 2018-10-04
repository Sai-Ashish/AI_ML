##############################################################################
#end the program using the keyboard interrupt
#keeps on asking the test data
##############################################################################

import numpy as np

#gate operation
def gate_mode(choice):
	if choice==1:
		return np.array([[1,0],[0,1],[1,1],[0,0]]),np.array([1,1,0,0])
	if choice==2:
		return np.array([[1,0],[0,1],[1,1],[0,0]]),np.array([0,0,1,0])
	if choice==3:
		return np.array([[1,0],[0,1],[1,1],[0,0]]),np.array([1,1,1,0])   

#generation of n data points
def generate_input(n,choice):
	X_src,y_src=gate_mode(choice)
	X=[]
	y=[]
	for i in range(0,4):
	    for j in range(0,n):
	        noise=np.random.normal(0,0.001,size=(2,1))
	        X.append(X_src[i].reshape(2,1)+noise)
	        y.append(y_src[i]+np.random.normal(0,0.001))

	#to maintain label balance +n data points will be added to maintain label class balance
	if choice==2 :#has more 0's than 1's in the src data set
	    for j in range(0,n):
	        noise=np.random.normal(0,0.001,size=(2,1))#adding some more 1 centered points
	        X.append(X_src[2].reshape(2,1)+noise)
	        y.append(y_src[2]+np.random.normal(0,0.001))
	if choice==3 :#has more 1's than 0's in the src data set
	    for j in range(0,n):
	        noise=np.random.normal(0,0.001,size=(2,1))#adding some more 0 centered points
	        X.append(X_src[3].reshape(2,1)+noise)
	        y.append(y_src[3]+np.random.normal(0,0.001))
	
	X=np.array(X)
	y=np.array(y)
	return X,y

# initialisation of the weights
def weights(no_nodes):

    W1=np.random.normal(0,1,(2,no_nodes))
    W2=np.random.normal(0,1,(no_nodes,1))
    b1=np.random.normal(0,1,(no_nodes,1))
    b2=np.random.normal(0,1,(1,1))
    return W1,W2,b1,b2

#layer  
def layer(w,x,b):
    return np.matmul(w.T,x.reshape(len(x),1))+b

# the sigmoid function as activation function
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

# derivative of sigmoid function
def derivative_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

#loss function
def mse(y_true,y_pred):
    return (y_true-y_pred)**2/len(y)

#forward path
def forward_path(W1,W2,b1,b2,x):
    z=sigmoid(layer(W1,x,b1))
    y_pred=sigmoid(layer(W2,z,b2))
    return z,y_pred

# back propagation
def back_propagation(delta_1,delta_2,bias_1,bias_2,y,
                                y_pred,z,X,W1,W2,b1,b2,no_nodes):
    layer1=layer(W1,X,b1)
    layer2=layer(W2,z,b2)
    bias_2 =bias_2-2.0*(y-y_pred)*derivative_sigmoid(layer2)
    delta_2=delta_2-2.0*(y-y_pred)*derivative_sigmoid(layer2)*z
    for j in range(0,no_nodes):
        bias_1[j]=bias_1[j]-2.0*(y-y_pred)*derivative_sigmoid(layer2)*derivative_sigmoid(layer1[j])*W2[j]
        delta_1[:,j]=delta_1[:,j]-2*(y-y_pred)*derivative_sigmoid(layer2)*W2[j]*derivative_sigmoid(layer1[j])*X
    return delta_1,delta_2,bias_1,bias_2

#test of the trained model
def test():
    while(1):    
        x_test=np.array((2,1), dtype='f')#input from user
        print('x_test[0]:')
        x_test[0]=float(input())
        print('x_test[1]:')
        x_test[1]=float(input())
        print (x_test)
        layer1= layer(W1,x_test,b1)
        z = sigmoid(layer1)
        layer2 = layer(W2,z,b2)
        y_pred= sigmoid(layer2)

        if y_pred>0.5:
        	print('Predicted probabilty of being class 1: '+str(y_pred))
        	print('Class: 1\n')
        else :
			print('Predicted probabilty of being class 0: '+str(1-y_pred))
			print('Class: 0\n')




###################################################################################################################
#main



print('Select gate operation:\n1)XOR\n2)AND\n3)OR')
choice=int(input())


#change the below number for changing the number of training data per set
no_train_samples_set = int(input("Enter the no of training samples per set: "))


no_nodes=int(input("Enter the no of nodes in hidden layer: "))

learning_rate=float(input("Value for learning rate: "))

X,y=generate_input(no_train_samples_set,choice)
X=X.reshape(len(y),2)
print('X:'+str(X.shape))
print('y:'+str(y.shape))

#weights initialisation
W1,W2,b1,b2=weights(no_nodes)

epochs=300
Loss=0

for k in range(0,epochs):
    
    delta_1 = np.zeros(W1.shape)
    delta_2 = np.zeros(W2.shape)
    bias_1  = np.zeros(b1.shape)
    bias_2  = np.zeros(b2.shape)
    Loss=0
    for i in range(0,len(y)):

        z,y_pred=forward_path(W1,W2,b1,b2,X[i])
        delta_1,delta_2,bias_1,bias_2=back_propagation(delta_1,delta_2,bias_1,bias_2,
                                                            y[i],y_pred,z,X[i],W1,W2,b1,b2,no_nodes)

        Loss=Loss+mse(y[i],y_pred)

    #print the loss in each epoch
    print('Epoch:'+str(k+1)+'         Loss:'+str(Loss))

    b2=b2-learning_rate*bias_2
    b1=b1-learning_rate*bias_1
    W2=W2-learning_rate*delta_2
    W1=W1-learning_rate*delta_1

print('\nTest:\n')
test()
