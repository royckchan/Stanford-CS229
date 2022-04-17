
import numpy as np
from sklearn.datasets import fetch_mldata
import time
import matplotlib.pyplot as plt

mnist = fetch_mldata('MNIST original')

X = mnist.data.astype(float)
Y = mnist.target.astype(float) 

mask = np.random.permutation(range(np.shape(X)[0]))

num_train = 10000
num_test = 500
K = 10

X_train = X[mask[:num_train]]
Y_train = Y[mask[:num_train]]

X_mean = np.mean(X_train,axis = 0)

X_train = (X_train-X_mean)/255

X_test = X[mask[num_train:num_train+num_test]]

X_test = (X_test - X_mean)/255

Y_test = Y[mask[num_train:num_train+num_test]]


print('X_train',X_train.shape)
print('Y_train',Y_train.shape)
print('X_test',X_test.shape)
print('Y_test',Y_test.shape)

ex_image = (np.reshape(X_train[10,:]*255 + X_mean, (28, 28))).astype(np.uint8)
plt.imshow(ex_image, interpolation='nearest')


# **Computing the distance matrix (num_test x num_train)**

# Version 1 (Naive implementation using two for loops)

start = time.time()
dists_1 = np.zeros((num_test,num_train))
for i in xrange(num_test):
    for j in xrange(num_train):
          dists_1[i,j] = np.sqrt(np.square(np.sum(X_test[i,:]-X_train[j,:])))

stop = time.time()
time_taken = stop-start
print('Time taken with two for loops: {}s'.format(time_taken))


# Version 2(Somewhat better implementation using one for loop)

start = time.time()
dists_2 = np.zeros((num_test,num_train))
for i in xrange(num_test):
          dists_2[i,:] = np.sqrt(np.square(np.sum(X_test[i,:]-X_train,axis = 1)))
        
stop = time.time()
time_taken = stop-start
print('Time taken with just one for loop: {}s'.format(time_taken))


# Version 3 (Fully vectorized implementation with no for loop)

start = time.time()
dists_3 = np.zeros((num_test,num_train))
A = np.sum(np.square(X_test),axis = 1)
B = np.sum(np.square(X_train),axis = 1)
C = np.dot(X_test,X_train.T)

dists_3 = np.sqrt(A[:,np.newaxis]+B[np.newaxis,:]-2*C)
        
stop = time.time()
time_taken = stop-start
print('Time taken with no for loops: {}s'.format(time_taken))

sorted_dist_indices = np.argsort(dists_3,axis = 1)

closest_k = Y_train[sorted_dist_indices][:,:K].astype(int)
Y_pred = np.zeros_like(Y_test)

for i in xrange(num_test):
      Y_pred[i] = np.argmax(np.bincount(closest_k[i,:]))


accuracy = (np.where(Y_test-Y_pred == 0)[0].size)/float(num_test)
print('Prediction accuracy: {}%'.format(accuracy*100))





