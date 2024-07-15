import torch
import torch.nn as nn
import math
import numpy as np

def accuracy(w, b, X, y):
    sigmoid = nn.Sigmoid()
    pred = sigmoid(w.T@X.T + b)
    pred = pred >= 0.5
    # print(pred)
    # pred = [1.0 if i else 0.0 for i in pred]
    comp = pred == y
    # print(comp)
    comp_ = [1.0 if i else 0.0 for i in comp[0]]

    # pred[pred>=0.5] = torch.full_like(pred[pred>=0.5], 1.0)
    # pred[pred<0.5] = torch.full_like(pred[pred<0.5], 0.0)

    # print(pred)
    # print("here:",pred[pred==y])
    
    # print(comp_)
    return np.mean(np.array(comp_)).item()

    # count = 0
    # for i in range(X.size(0)):
    #     val = sigmoid(torch.matmul(w.T, X[i]) + b)
    #     pred = 0
    #     if (val >= 0.5):
    #         pred = 1
    
    #     if (y[i]==pred):
    #         count += 1
        
        
    # return count/X.size(0)

def out(s1, s2, m, z):
    result = torch.sub(s1, s2)
    # result /= (2*m)
    # result = result.item() * z
    result = result * z/(2*m)
    return result

def error_fn(y_hat, y):
    # if torch.all(y_hat == torch.ones_like(y)):
    #     y_hat = torch.full_like(y, 0.999999999)
    # if torch.all(y_hat == torch.zeros_like(y)):
    #     y_hat = torch.full_like(y, 0.000000001)

    # if (y_hat == 1):
    #     y_hat = 0.999999999
    # if (y_hat == 0):
    #     y_hat = 0.000000001
    y = y.numpy()
    y_hat = y_hat.numpy()
    # print(y_hat[0] == 1)
    # print(y_hat[0][y_hat[0] == 1])

    y_hat[0] = np.clip(y_hat[0], 0.000001, 0.999999)
    # print(y_hat) 
    # y_hat[(y_hat == 1)] = torch.full_like(y_hat[(y_hat == 1)], 0.999999999)
    # print(y_hat.shape)
    # y_hat[(y_hat == 0)] = torch.full_like(y_hat[(y_hat == 0)], 0.000000001)
    # print(y_hat)

    return np.mean(-y*np.log(y_hat) - (1-y)*np.log(1 - y_hat))

def get_grad(w, b, X, y, m = 0.0001):
    d_w = w.size()[0] # 196608
    d_b = b.size()[0] # 1
    z_w = torch.randn(d_w, 1)
    z_b = torch.randn(d_b, 1)
    final_grad_w = torch.zeros(d_w, 1)
    final_grad_b = torch.zeros(d_b, 1)

    n = X.size(0)

    sigmoid = nn.Sigmoid()

    # Weights
    mm1 = torch.matmul( (w+m*z_w).T , X.T)
    # print(mm1.shape)
    # print(mm1.size()) # [1, 217]
    # mm1 = torch.matmul( (w+m*z_w) , X)
    s1 = sigmoid(mm1 + b)
    # print(s1.shape)

    mm2 = torch.matmul( (w-m*z_w).T , X.T)
    # mm2 = torch.matmul( (w-m*z_w) , X)
    s2 = sigmoid(mm2 + b)

    # print("error:", error_fn(s1, y).size()) # torch.Size([1, 217])
    # print("out:", out(error_fn(s1, y), error_fn(s2, y), m, z_w).size()) # torch.Size([196608, 217])
    # final_grad_w += out(error_fn(s1, y), error_fn(s2, y), m, z_w)

    final_grad_w = out(error_fn(s1, y), error_fn(s2, y), m, z_w)
    # print(error_fn(s1, y))
    # print(error_fn(s1, y).shape)
    # print(temp.shape)
    # final_grad_w = torch.mean(temp, 1)
    # print("final_grad_w:", final_grad_w.size())
    
    # Bias
    mm1 = torch.matmul(w.T, X.T)
    mm2 = m*z_b
    s1 = sigmoid(mm1 + b + mm2)
    s2 = sigmoid(mm1 + b - mm2)
    
    # print("error:", error_fn(s1, y).size()) # torch.Size([1, 217])
    # print("out:", out(error_fn(s1, y), error_fn(s2, y), m, z_b).size()) # torch.Size([1, 217])
    temp = out(error_fn(s1, y), error_fn(s2, y), m, z_b)
    final_grad_b = torch.sum(temp, 1)
    # print("final_grad_b:", final_grad_b.size()) # torch.Size([1])

    # for i in range(n):
    #     sigmoid = nn.Sigmoid()

    #     # Weights
    #     # print("X:", X.shape)
    #     # print("w:", w.shape)
    #     # print("z_w:", z_w.shape)
    #     # print("X[i]:", torch.unsqueeze(X[i],1))
    #     mm1 = torch.matmul( (w+m*z_w).T , torch.unsqueeze(X[i],1))
    #     # print(mm1 + b)
    #     s1 = sigmoid(mm1 + b)

    #     mm2 = torch.matmul( (w-m*z_w).T , torch.unsqueeze(X[i],1))
    #     s2 = sigmoid(mm2 + b)
    #     # print(type(s1.item()), type(s2.item()))
    #     final_grad_w += out(error_fn(s1, y[i]), error_fn(s2, y[i]), m, z_w)

    #     # Bias
    #     mm1 = torch.matmul(w.T, torch.unsqueeze(X[i],1))
    #     mm2 = m*z_b
    #     s1 = sigmoid(mm1 + b + mm2)
    #     s2 = sigmoid(mm1 + b - mm2)
        
    #     final_grad_b += out(error_fn(s1, y[i]), error_fn(s2, y[i]), m, z_b)

    return final_grad_w, final_grad_b