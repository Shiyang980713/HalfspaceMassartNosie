import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal as mvn
from sklearn.preprocessing import MinMaxScaler, normalize
from visdom import Visdom



def GTdata(num_data, d_data):
    #Ground Truth Data
    num_data = int(num_data/2)#每一类的样本数量
    n_d_data = torch.ones(num_data,d_data)

    x1 = torch.normal(3*n_d_data, 1.5)#mean = 3,sigma = 1.5     ｜｜+1类
    y1 = torch.ones(num_data)

    xm1 = torch.normal(-3*n_d_data, 1.5)#mean = -3,sigma = 1.5  ｜｜-1类
    ym1 = -torch.ones(num_data)
    #w^* = [1, 1, 0]
    #L2norm -> [(1/2)^(1/2), (1/2)^(1/2), 0]

    x_g = torch.cat([x1,xm1],0).type(torch.FloatTensor)
    y_g = torch.cat([y1,ym1]).type(torch.LongTensor)
    
    return x_g, y_g


#update data with Massart Noise
def AddMassart(x, y, eta):
    #Define Massart Noise as UniNormal
    #The result is not limitted by this assumption
    mean = np.array([0, 0])
    cov = np.matrix([[30, 10],[10, 30]])
    noise = mvn(mean=mean, cov=cov)

    y = y.numpy()#noise label
    y_n = np.zeros(shape=y.shape)#noise label

    p_noise = noise.cdf(abs(x))#eta(x)
    
    for i in range(len(y)):
        if  p_noise[i] < eta:
            #p = eta(x) inverse
            y_n[i] = -y[i]
        else:
            #p = 1 - eta(x) hold
            y_n[i] = y[i]
    
    y_n = torch.tensor(y_n).type(torch.LongTensor)
    return y_n

#scale the feature: ||x||2 <= 1
def preprocess(x):
    x = x.numpy()
    x_norm = np.linalg.norm(x, ord=2, axis=1)
    #x_norm_max = x_norm.max()
    #x_norm_min = x_norm.min()
    #x_norm = x_norm[:,None]
    x_norm = np.expand_dims(x_norm, axis=1)#(1000,)->(1000,1)
    mm = MinMaxScaler()
    scaled_x_norm = mm.fit_transform(x_norm)
    scaler = scaled_x_norm / x_norm
    scaler = np.squeeze(scaler)#(1000,1)->(1000,)
    normed_x = x.T * scaler
    normed_x = normed_x.T

    normed_x = torch.tensor(normed_x).type(torch.FloatTensor)
    return normed_x

def sample_with_margin(x,y,num_sample_data,gamma):
    x = x.numpy()
    y = y.numpy()
    x_s = []
    y_s = []
    
    #optimal w(calculate by prior set)
    w_s = np.array([1,1,0])
    w_s = np.expand_dims(w_s, axis=0)
    w_s = normalize(w_s, norm='l2')
    w_s = np.squeeze(w_s, axis=0)
    #delete point in [-gamma,gamma]
    for i in range(len(y)):
        inprod = np.inner(x[i],w_s[:-1]) + w_s[-1]
        if abs(inprod) < gamma:
            continue
        else:
            x_s.append(x[i])#(n,2)
            y_s.append(y[i])#(n,1)

    #choose some point
    y_s = np.expand_dims(y_s, axis=1)#(n,)->(n,1)
    data_arr = np.concatenate((x_s,y_s),axis = 1)#(n,2)|(n,1)->(n,3)
    row_total = data_arr.shape[0]
    row_sequence= np.random.choice(
        row_total,
        num_sample_data,
        replace=False
        )
    data_arr = data_arr[row_sequence,:]

    x_s = data_arr[:,:-1]
    y_s = data_arr[:,-1]

    x_s = torch.tensor(x_s).type(torch.FloatTensor)
    y_s = torch.tensor(y_s).type(torch.LongTensor)
    return x_s, y_s

def resample_data(x,y,num_sample_data):
    x = x.numpy()
    y = y.numpy()
    x_s = x
    y_s = y
    
    #choose some point
    #y_s = np.expand_dims(y_s, axis=1)#(n,)->(n,1)
    data_arr = np.concatenate((x_s,y_s),axis = 1)#(n,2)|(n,1)->(n,3)
    row_total = data_arr.shape[0]
    row_sequence= np.random.choice(
        row_total,
        num_sample_data,
        replace=True
        )
    data_arr = data_arr[row_sequence,:]

    x_s = data_arr[:,:-1]
    y_s = data_arr[:,-1]
    y_s = np.expand_dims(y_s, axis=1)

    x_s = torch.tensor(x_s).type(torch.FloatTensor)
    y_s = torch.tensor(y_s).type(torch.LongTensor)
    x_s = x_s.cpu()
    y_s = y_s.cpu()
    return x_s, y_s




'''
plt.scatter(x_s.data.numpy()[:,0],x_s.data.numpy()[:,1],
            s = 10, c = y_s.data.numpy(), cmap = 'coolwarm')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.show()
'''



