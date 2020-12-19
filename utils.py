import numpy as np
import torch
from matplotlib import pyplot as plt



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






