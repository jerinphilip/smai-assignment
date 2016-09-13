import numpy as np

m, n, p = 3, 2, 1
#W21 = np.ones((n, m))
#W32 = np.ones((p, n))
W21 = np.random.randn(n, m)
W32 = np.random.randn(p, n)
sigmoid = lambda x: 1.0/(1.0+np.exp(-x))
ds = lambda x: np.multiply(sigmoid(x), (1.0-sigmoid(x)))
#ds = lambda x: sigmoid(x) * (1-sigmoid(x))
eta = 1e1

ips = [[0, 0], [0, 1], [1, 0], [1, 1]]
for i in range(100000):
    print("Iter", i)
    for ip in ips:
        x = np.array(ip).astype(np.float_)
        x_ = np.append([1], x)
        #x_ = x

        netj = W21.dot(x_.T)
        y = sigmoid(netj)
        netk = W32.dot(y.T)
        z = sigmoid(netk)
        #print("netj", netj)
        #print("y", y)
        #print("netk", netk)
        #print("z", z)
        print(ip, "->", z)

        #print("Backpropogation")
        op = np.array([(x[0] and x[1])*1.0])
        #print("df(netk", ds(netk))
        df_netk = ds(netk)
        #print("dk: ", dk)
        dk = np.multiply(-(op-z), df_netk)

        dW = np.zeros((p, n))
        for k in range(p):
            for j in range(n):
                dW[k, j] = -(op[k]-z[k])* df_netk[k] * y[j]
        #print("dW: ", dW)

        W32 = W32 - eta*dW
        #print("W32:", W32)

        #wTd = np.zeros(n)
        #for k in range(p):
        #    for j in range(n):
        #        wTd[j] += -(op[k]-z[k]) * df_netk[k] * W32[k, j]


        df_netj = ds(netj)
        #print("wTd:", wTd)
        dW2 = np.zeros((n, m))
        for j in range(n):
            for i in range(m):
                for k in range(p):
                    dW2[j, i] += -(op[k]-z[k])*(df_netk[k]) * W32[k, j] * df_netj[j] * x_[i]

        #print("dW2", dW2)
        W21 = W21 - eta*dW2
        #print("W21:", W21)
        #s = input()


