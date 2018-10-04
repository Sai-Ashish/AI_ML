
import numpy as np
import cvxpy as cp
import csv
from sklearn import svm




# Import Data
def read_data(filename):
    X=[]
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            X.append(np.array(row).astype('float64'))
    X=np.array(X)
    return X

# Convex optimization
def convex_solve(X,Y,var):

    R = cp.norm(cp.matmul(X.T,cp.matmul(cp.diag(var),Y)))**2
    R.shape

    P = cp.sum(var)
    Const1 = P - 0.5*R

    Const2 = [0<=var,cp.matmul(var.T,Y) == 0]
    obj = cp.Maximize(Const1)
    prob = cp.Problem(obj, Const2)
    prob.solve(verbose=True)

    return var.value

#find the weights from the lagrangian multiplier
def weights(var,X,Y):

    W=np.dot(X.T,np.matmul(np.diag(var),Y))
    W0 = (1/Y[281]) - np.dot(W.T,X[281])#281 index corresponds to the non zero lagrangian multiplier
    return W0,W

#verification by svm from sklearn
def verify(X,Y,Test):
    clf = svm.SVC()
    clf.fit(X,Y.reshape(len(Y),))
    pred=clf.predict(Test)
    print (pred)

#TEST
def test(W0,W,X,Y):
    X_test=np.array([[2,0.5],[0.8,0.7],[1.58,1.33],[0.008, 0.001]])
    for elem in X_test:
        if np.dot(W.T,elem)+W0>0:
            print(str(elem)+"\nBelongs to class 1")
        else:
            print(str(elem)+"\nBelongs to class -1")
    
    print("verification by svm from sklearn")
    verify(X,Y,X_test)


#IMPORT DATA
X=read_data('Xsvm.csv')
Y=read_data('ysvm.csv')

var = cp.Variable(len(Y))
#FIND THE VARIABLES BY CONVEX OPTIMIZER
var=convex_solve(X,Y,var)

#FIND THE WEIGHTS FROM THE LAGRANGIAN MULTIPLIERS
W0,W=weights(var,X,Y)
print ('W0:'+str(W0)+'\nW:'+str(W)+'\n')

#testing
print('Test:\n')
test(W0,W,X,Y)
