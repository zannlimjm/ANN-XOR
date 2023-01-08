import numpy as np
import matplotlib.pyplot as plt

# These are XOR inputs
x=np.array([[0,0,1,1],[0,1,0,1]])
# These are XOR outputs
y=np.array([[0,1,1,0]])
# Number of inputs
n_x = 2
# Number of neurons in output layer
n_y = 1
# Number of neurons in hidden layer
n_h = 3
# Total training examples
m = x.shape[1]
# Learning rate
lr = 0.01
# Define random seed for consistent results
np.random.seed(2)
# Define weight matrices for neural network
w1 = np.random.rand(n_h,n_x)   # Weight matrix for hidden layer
w2 = np.random.rand(n_y,n_h)   # Weight matrix for output layer
# w1 = np.zeros((n_h,n_x))
# w2 = np.zeros((n_y,n_h))
# print(w1)
# print(w2)
# no bias units are used for this assignment
# Initalize ist to accumulate losses
losses = []
y1_loss = []
y2_loss = []
y3_loss = []
y4_loss = []

def sigmoid(z):
    z= 1/(1+np.exp(-z))
    return z

# Forward propagation
def forward_prop(w1,w2,x):
    v1 = np.dot(w1,x)
    a1 = sigmoid(v1)    
    v2 = np.dot(w2,a1)
    a2 = sigmoid(v2)
    return v1,a1,v2,a2

# Backward propagation
def back_prop(m,w1,w2,z1,a1,z2,a2,y):
    
    dz2 = a2-y
    dw2 = np.dot(dz2,a1.T)/m
    dz1 = np.dot(w2.T,dz2) * a1*(1-a1)
    dw1 = np.dot(dz1,x.T)/m
    dw1 = np.reshape(dw1,w1.shape)
    
    dw2 = np.reshape(dw2,w2.shape)    
    return dz2,dw2,dz1,dw1
epochs = 10000
y1 = []
y2 = []
y3 = []
y4 = []
for i in range(epochs):
    z1,a1,z2,a2 = forward_prop(w1,w2,x)
    #Predicted Output
    y1.append(a2[0][0])
    y2.append(a2[0][1])
    y3.append(a2[0][2])
    y4.append(a2[0][3]) 
    # loss = -(1/m)*np.sum(y*np.log(a2)+(1-y)*np.log(1-a2))
    loss = 0.5 * (y - a2) ** 2
    losses.append(np.sum(loss))
    y1_loss.append(loss[0][0])
    y2_loss.append(loss[0][1])
    y3_loss.append(loss[0][2])
    y4_loss.append(loss[0][3])
    # losses.append(loss)
    da2,dw2,dz1,dw1 = back_prop(m,w1,w2,z1,a1,z2,a2,y)
    w2 = w2-lr*dw2
    w1 = w1-lr*dw1



# Plot losses to check network performance (if converges)
plt.plot(losses)
plt.xlabel("EPOCHS")
plt.ylabel("Loss value")
plt.show()

#plotting individual inputs
plt.plot(y1_loss)
# plt.plot(y1)
plt.xlabel("EPOCHS")
# plt.ylabel("Predicted output for input [1,1]")
plt.ylabel("Loss value for input [1,1]")
plt.show()

plt.plot(y2_loss)
# plt.plot(y2)
plt.xlabel("EPOCHS")
# plt.ylabel("Predicted output for input [0,1]")
plt.ylabel("Loss value for input [0,1]")
plt.show()


plt.plot(y3_loss)
# plt.plot(y3)
plt.xlabel("EPOCHS")
# plt.ylabel("Predicted output for input [1,0]")
plt.ylabel("Loss value for input [1,0]")
plt.show()

plt.plot(y4_loss)
# plt.plot(y4)
plt.xlabel("EPOCHS")
# plt.ylabel("Predicted output for input [0,0]")
plt.ylabel("Loss value for input [0,0]")
plt.show()
