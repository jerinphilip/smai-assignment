import numpy as np

W21 = np.ones((3, 2))
W32 = np.ones((2, 3))
sigmoid = lambda x: 1/(1+np.exp(-x))
ds = lambda x: np.multiply(sigmoid(x), (1-sigmoid(x)))

x = np.array([1, 2], dtype='float32')
netj = W21.dot(x.T)
y = sigmoid(netj)
netk = W32.dot(y.T)
z = sigmoid(netk)
print("netj", netj)
print("y", y)
print("netk", netk)
print("z", z)

print("Backpropogation")
op = np.array([0.52, 0.53])
print("df(netk", ds(netk))
dk = np.multiply(-(op-z) , ds(netk))
print("dk: ", dk)

dW = np.zeros((2, 3))
for k in range(2):
    for j in range(3):
        dW[k, j] = dk[k] * y[j]
print("dW: ", dW)

wTd = np.zeros(3)
for k in range(2):
    for j in range(3):
        wTd[j] += dk[k] * W32[k, j]

W32 = W32 - 1e1*dW
print("W32:", W32)

df_netj = ds(netj)
print("wTd:", wTd)
dW2 = np.zeros((3, 2))
for j in range(3):
    for i in range(2):
        dW2[j, i] = wTd[j] * df_netj[j] * x[i]

print("dW2", dW2)
W21 = W21 - 1e1*dW2
print("W21:", W21)

