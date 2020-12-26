import numpy as np
import torch
from matplotlib import pyplot as plt
from MakeData import *

def test_model(D, num_test_data):
    x_tt, y_tt = GTdata(num_test_data, 2)
    #scale
    x_t = preprocess(x_tt).numpy()
    y_t = y_tt.numpy()
    y_p = []

    for i in range(len(y_t)):
        if abs(np.inner(x_t[i], D[0][0])) >= D[0][1]:
            y_p.append(np.sign(np.inner(x_t[i], D[0][0])))
            continue
        if abs(np.inner(x_t[i], D[1][0])) >= D[1][1]:
            y_p.append(np.sign(np.inner(x_t[i], D[1][0])))
            continue
        y_p.append(np.sign(np.inner(x_t[i], D[2][0])))
        # if abs(np.inner(x_t[i], D[2][0])) >= D[2][1]:
        #     y_p.append(np.sign(np.inner(x_t[i], D[2][0])))
        #     continue

    dis = y_p - y_t
    true = dis.tolist().count(0)
    error_rate = 1 - true/len(dis)

    return error_rate



def show_data(x, y, xlim, ylim, cc = False, ci = False):
    plt.figure(num=1,figsize=(5,5))
    if cc == True:
        plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],
                s = 1, c = y.data.numpy(), cmap = 'coolwarm')
        plt.Circle((0,0),0.5)
    else:
        plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],
                s = 1, c = 'g')

    if ci == True:
        circle1=plt.Circle((0,0),1, color='r',fill = False)
        plt.gcf().gca().add_artist(circle1)
    
    ax = plt.gca()
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))
    plt.xlim(-xlim,xlim)
    plt.ylim(-ylim,ylim)
    plt.show()

def line_on_data(w):


    plt.figure(num='line on data',figsize=(5,5))






