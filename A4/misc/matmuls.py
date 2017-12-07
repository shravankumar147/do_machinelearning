import numpy as np
def sigmoid(x):
	return 1/(1+np.exp(-x))

def dsig(x):
	return (1-x)*x
Xi = np.array([[4,3,0]])
x = Xi
d = x.shape[1]
nH = 3
c = 2
lrate = 0.1
# x = np.c_[Xi,np.ones(Xi.shape[0])]	
print(x)
# W1 = np.array([[1,1,1],[3,3,3],[2,2,2]]).astype('float')
W1 = np.random.uniform(0,1,(d,nH))
wji = W1
# wji = np.r_[W1,np.ones((1,W1.shape[1]))]
print(W1.shape, wji.shape)
W2 = np.ones((nH,c))
wkj = W2
# wkj = np.r_[W2,np.ones((1,W2.shape[1]))]
print(W2.shape, wkj.shape)

b1 = 1
b2 = 1
tk = np.array([1,2])
for iter in xrange(10000):
	print("iteration{}".format(iter))
	# input layer 
	# size of x is 1x(d+1), additional term for bais input
	# x = np.random.random_sample((1,4))
	# x = np.array([[4,3,2,1]])
	# print(x,x.shape)
	# wieght to comput netj. 
	# size of wji is (d+1)xnH
	# wji = np.random.uniform(0,1,(4,3))
	# wji = np.array([[1,1,1],[3,3,3],[2,2,2],[0,0,0]])
	# print(wji, wji.shape)
	# size of netj( 1xnH)
	netj = np.dot(x,wji)+b1
	# print(netj)




	# output at hidden layer and input to output layer
	# size is same as netj i.e. 1xnH
	yj = sigmoid(netj)
	# print(yj)
	# wieghts at hidden to output layer to compute netk
	# wkj size = (nH+1)xc
	
	# print("Hidden to Outpur net")
	# print(yj.shape, wkj.shape, yj.shape[1])

	netk = np.dot(yj,wkj) + b2
	zk = sigmoid(netk)
	# print(netk, zk)
	# print(netk.shape, zk.shape)

	# tk = np.ones((1,2))
	err = (tk-zk)
	# print(err)

	dzk = dsig(zk)
	# print(dzk)
	deltaK = -err*dzk
	# print(deltaK)	

	# use in backpropagation from output laye to hidden layer 
	a = deltaK # sensitivities computed at output layer deltak
	b2 = a
	b = yj # outputs of hidden units yj,bais term concatinated as bottom row (since bias update not affected by yj's)
	# print(a.shape, b.shape)
	Wkj = np.dot(a.T,b).T # computing the gradient of error with respect to wkj
	# update the weights wkj
	wkj += -lrate*Wkj
	# print(wkj.shape)

	# print(yj)
	# print(wkj)

	## computing senstivity for each hidden node

	# print(deltaK.shape, wkj.shape)

	S = np.dot(deltaK,wkj.T)
	dyj = dsig(yj)

	deltaJ = dyj*S
	
	a = deltaJ
	b1 = a
	b = x
	Wji = np.dot(a.T,b).T
	wji += -lrate*Wji
	# print(wji.shape)
	print(zk)
