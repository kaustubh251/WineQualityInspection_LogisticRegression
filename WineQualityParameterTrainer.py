import numpy as npy
file = open('RedWineQuality.csv','rt')
data = file.read()
dataSp = data.split()
N = len(dataSp)
print(N)
dataS = dataSp[9:(N-1)]
npy.random.shuffle(dataS)
dataXtrain = []
dataYtrain = []
dataXvalid = []
dataYvalid = []
dataXtest = []
dataYtest = []
for i in range(N-10):
    if i<1000:
        dataSrow = dataS[i]
        Xrow = dataSrow.split(',')
        XrowFloat = [1]
        for j in range(len(Xrow)-1):
            XrowFloat.append(float(Xrow[j]))
        dataXtrain.insert(i, XrowFloat)
        y = float(Xrow[len(Xrow) - 1])
        if y>7:
            dataYtrain.append(1)
        if y<=7:
            dataYtrain.append(0)
    counter = 0
    if (i>=1000)and(i<1300):
        dataSrow = dataS[i]
        Xrow = dataSrow.split(',')
        XrowFloat = [1]
        for j in range(len(Xrow)-1):
            XrowFloat.append(float(Xrow[j]))
        dataXvalid.insert(counter, XrowFloat)
        y = float(Xrow[len(Xrow) - 1])
        if y>7:
            dataYvalid.append(1)
        if y<=7:
            dataYvalid.append(0)
        counter += 1
    counter = 0
    if (i>=1300):
        dataSrow = dataS[i]
        Xrow = dataSrow.split(',')
        XrowFloat = [1]
        for j in range(len(Xrow)-1):
            XrowFloat.append(float(Xrow[j]))
        dataXtest.insert(counter, XrowFloat)
        y = float(Xrow[len(Xrow) - 1])
        if y>7:
            dataYtest.append(1)
        if y<=7:
            dataYtest.append(0)
        counter += 1

def z(theta, x1):
    Z = 0
    for i in range(len(x1)):
        Z += theta[i]*x1[i]
    return Z

def h(Z):
    return 1/(1 + npy.exp(-Z))

def cost(theta, x, y):
    cost = 0
    for i in range(len(x)):
        x1 = x[i][:]
        Z = z(theta, x1)
        H = h(Z)
        cost += -(y[i]*npy.log(H) + ((1-y[i])*npy.log(1-H)))
    return cost/len(x)

def gradDescent(theta, x, y, alpha, maxIter, regParam):
    for k in range(maxIter):
        theta1 = theta
        for i in range(len(theta)):
            derCost = 0
            j = 0
            for j in range(len(x)):
                x1 = x[j][:]
                Z = z(theta, x1)
                H = h(Z)
                derCost += (H - y[j])*x[j][i] + regParam*theta[i]
            theta1[i] = theta[i] - alpha*derCost/len(x)      
        theta = theta1
    return theta

def accuracy(theta_Final, x, y):
    q = 0
    p = []
    for i in range(len(x)):
        x1 = x[i][:]
        Z = z(theta_Final, x1)
        H = h(Z)
        if H>=0.5:
            p.insert(i, 1)
        if H<0.5:
            p.insert(i, 0)
        if y[i]==p[i]:
            q += 1
    accuracy = q*100/len(y)
    return accuracy

init_Theta = []
for i in range(len(dataXtrain[0])):
    init_Theta.append(1)

alpha = 0.03
maxIter = 1000
regParam = 1
theta_Final = gradDescent(init_Theta, dataXtrain, dataYtrain, alpha, maxIter, regParam)
print(theta_Final)
print(accuracy(theta_Final, dataXtrain, dataYtrain))
print(accuracy(theta_Final, dataXvalid, dataYvalid))

