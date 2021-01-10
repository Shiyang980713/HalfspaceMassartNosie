import os
import sklearn
import numpy as np
import torch
from matplotlib import pyplot as plt

import net
from MakeData import *
from utils import *

def loss_func(x, y, lamb):
    z = -y * x
    for i in range(len(z)):
        if z[i] >= 0:
            z[i] = (1 - lamb) * z[i]
            #LLR.append((1 - lamb) * z[i])
        else:
            z[i] = lamb * z[i]
            #LLR.append(lamb * z[i])
    loss = torch.mean(z)
    return loss
    
'''
def stop_sign():

'''

def findT(w, x, y, gamma, eps):
    percent = gamma * eps
    inprod = []
    jump = 0
    
    for i in range(len(y)):
        inprod.append(torch.mm(w, x[i].view(-1,1)))

    inprod = torch.cat(inprod, dim=0)

    #(I)make sure P( <w,x> >= T) >= gamma*eps
    #find the upper bound of T
    #计算当前决策面w下内积最大的样本列表
    top_n = int(len(y) * percent)
    # if top_n == 0:
    #     jump = 1
    #     top_n = int(len(y)/2)
    upper_list = torch.topk(torch.abs(inprod), top_n, dim=0)
    upper_T,_ = upper_list
    #T应该小于这些内积中的最小值
    T_max = torch.min(upper_T)

    #(II)make sure min P( h(x) != y | <w,x> >= T)
    #find the lowerbound of T, 
    #计算当前决策面w下, 分类错误的内积列表
    n_error = 0
    error_list = []
    n_right = 0
    for i in range(len(y)):
        if torch.sign(inprod[i]) == y[i]:
            n_right += 1
        else:
            n_error += 1
            error_list.append(torch.abs(inprod[i]))
    error_list = torch.cat(error_list, dim=0)
    #T应该大于这些内积中的最大值
    T_min = torch.max(error_list)    
    
    #T = min(max_T, min_T)
    T_i = min(T_max, T_min)
    # if jump == 1:
    #     T_i = T_max
    w_i = w

    return w_i, T_i

def update_region(x, y, w, T):
    x_s = []
    y_s = []
    for i in range(len(y)):
        if abs(torch.mm(w, x[i].view(-1,1)))  >= T:
            continue
        else:
            x_s.append(x[i])
            y_s.append(y[i]) 
    x_s = torch.stack(x_s)
    y_s = torch.stack(y_s)

    x_s = x_s.cpu()
    y_s = y_s.cpu()
    return x_s, y_s

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='2'

    #random seed
    torch.manual_seed(713)
    torch.cuda.manual_seed_all(713)
    np.random.seed(713)
    
    #PAC Parameter
    gamma = 0.1#Margin 0.1
    eta = 0.417#eta Upper Bound0.417
    eps = 0.09#precision
    lamb = eta + eps#LeakyReLU lamb ~= eta

    #Data Parameter
    num_data = 10000#10000
    num_sample_data = 1000#1000
    d_data = 2

    #Train Parameter
    EPOCH = 2#SGD epoch
    LenD = 0

    #(x_g,y_g)   GT data
    x_g, y_g = GTdata(num_data,d_data)

    #(x_n,y_n)   加入标签噪声
    y_n = AddMassart(x_g, y_g, eta)
    x_n = x_g

    #(x_p,y_p)   数据scale||x||2 <= 1
    x_p = preprocess(x_n)
    y_p = y_n

    #(x_s,y_s)sampled large margin
    x_s, y_s = sample_with_margin(x_p, y_p, num_sample_data, gamma)
    y_s = torch.unsqueeze(y_s, 1)

    #plot
    # show_data(x_g,y_g,10,10)
    #show_data(x_g,y_g,10,10,cc=True)
    #show_data(x_n,y_n,10,10,cc=True)
    # show_data(x_p,y_p,1,1,cc=True)
    #show_data(x_s,y_s,1.2,1.2,cc=True,ci=True)

    #Decision list
    D = []
    LEFT = len(y_s)


    while len(y_s) >= num_sample_data * eps:
        LenD += 1
        print('loop:{}'.format(LenD),'num_sample:{}'.format(len(y_s)))

        #load data

        dataset = torch.utils.data.TensorDataset(x_s, y_s)
        dataloader = torch.utils.data.DataLoader(
                                                dataset,
                                                batch_size=num_sample_data,
                                                #如果分小批，要把每一批的loss取平均
                                                shuffle=True,
                                                num_workers=8,
                                                pin_memory=True
        )

        model = net.MNet(d_data)
        model.cuda()
        opt = torch.optim.SGD(model.parameters(), lr=1e-3)
        model.train()

        for epoch in range(EPOCH):
            running_loss = 0.0
            for i, data in enumerate(dataloader, 0):
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()
        
                # zero the parameter gradients
                opt.zero_grad()
                
                # forward + backward + optimize
                outputs = model(inputs)
                loss = loss_func(outputs, labels, lamb)
                loss.backward()                
                opt.step()

                #constrain the w
                #！！！考虑w初始值为0的情况！！！
                for p in model.parameters():
                    w = p.data
                    norm_w = torch.norm(w, p=2)
                    p.data = torch.div(p.data, norm_w)
        
                running_loss += loss.item()
            
            #print loss/ w_n / norm & save w_n
            for p in model.parameters():
                w_n = p.data
            #print('w is :{}||'.format(w_n),'her norm :{}||'.format(torch.norm(w_n, p=2)))
            print('Loss: {}'.format(running_loss))

        x_c, y_c = resample_data(x_s, y_s,500)
        x_s = x_s.cuda()
        y_s = y_s.cuda()
        x_c = x_c.cuda()
        y_c = y_c.cuda()
        w_i, T_i = findT(w_n, x_c, y_c, gamma, eps)

        x_s, y_s = update_region(x_s, y_s, w_i, T_i)
        #x_s, y_s = resample_data(x_s, y_s,500)
        '''
        x_s = x_s.cuda()
        y_s = y_s.cuda()
        w_i, T_i = findT(w_n, x_s, y_s, gamma, eps)
        x_s, y_s = update_region(x_s, y_s, w_i, T_i)
        '''

        w_i = w_i.cpu()
        T_i = T_i.cpu()
        print('w_{} = ({})'.format(LenD, w_i.data),
                'T_{} = {}'.format(LenD, T_i.data)
        )
        print(LEFT)
        w_i = torch.squeeze(w_i)
        D.append([w_i.numpy().tolist(),T_i.numpy().tolist()])
        

    print("All data are classified, Decision list is:")
    print(D)
    err  = test_model(D, 200)
    print("err is:")
    print(err)




    






