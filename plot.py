from matplotlib import pyplot as plt
import numpy as np

gamma = 0.1#Margin
eta = 0.417#eta Upper Bound
eps = 0.1#precision
lamb = 0.2#LeakyReLU lamb ~= eta

def FLeakyReLU(x, lamb):
    y = [] 
    for i in range(len(x)):
        if x[i] >= 0:
            y.append((1 - lamb) * x[i])
            #LLR.append((1 - lamb) * z[i])
        else:
            y.append(lamb * x[i])
            #LLR.append(lamb * z[i])
    return y

def Fsign(x):
    y = [] 
    for i in range(len(x)):
        if x[i] >= 0:
            y.append(1)
        else:
            y.append(0)
    return y

x = np.linspace(-5, 5, 500)
y1 = FLeakyReLU(x, lamb)
y2 = Fsign(x)

plt.figure(num=1,figsize=(3,3))
plt.plot(x, y1)
ax = plt.gca()
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))
plt.xlim(-4,4)
plt.ylim(-4,4)


plt.figure(num=2,figsize=(3,3))
plt.plot(x, y2)
ax = plt.gca()
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))
plt.xlim(-4,4)
plt.ylim(-4,4)
plt.show()




