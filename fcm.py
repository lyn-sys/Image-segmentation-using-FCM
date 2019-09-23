import cv2
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(1234)#Random seeds, random to the same array at a time
def fcm(X,k,b):
    iter = 0
    N,p = X.shape

    P = np.random.randn(N,k)#Generating Random Numbers with Normal Distribution

    P = P / np.dot(np.sum(P,1).reshape(N,1),np.ones((1,k)))

    J_prev = np.inf
    J = []

    while True:
        t = pow(P,b)
        #Here's a concrete iteration algorithm
        C = np.dot(X.T, t).T / (sum(t, 0).reshape(k, 1) * np.ones((1, p)))

        dist = np.dot(np.sum(C*C,1).reshape(k,1),np.ones((1,N))).T+\
               np.sum(X*X,1).reshape(N,1)*np.ones((1, k)) - \
               2*np.dot(X,C.T)

        t2 = pow(1.0/dist,1.0/(b-1))
        P = t2/(np.sum(t2,1).reshape(N,1)*np.ones((1,k)))
        J_cur = sum(sum((pow(P,b))*dist,0),0)/ N
        J.append(J_cur)

        print(iter, J_cur)
        if abs(J_cur - J_prev) < 0.001:
            #Stop iteration when the error after iteration is less than 0.001
            break

        iter += 1

        J_prev = J_cur
    return C,dist,J
img = cv2.imread('image-sonar/sonar5.png')
m,n,p = img.shape
img = img.astype(np.double)#This step is critical for converting to double type
img = img.reshape(m*n,p)
C,dist,J = fcm(img,3,2)
#J is the error value of each iteration, which can be displayed by plot.
label = dist.argmin(axis=1)
print(label.shape)
img_1 = C[label,:]
img_2 = img_1.reshape(m,n,p)
plt.imshow(img_2/255,cmap='gray')
plt.show()
