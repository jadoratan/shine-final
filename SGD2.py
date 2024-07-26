import torch
import torch.nn as nn
import math
import numpy as np

def accuracy(w, b, X, y):
    sigmoid = nn.Sigmoid()
    pred = sigmoid(w.T@X.T + b)
    pred = pred >= 0.5
    comp = pred == y
    comp_ = [1.0 if i else 0.0 for i in comp[0]]
    
    return np.mean(np.array(comp_)).item()

def out(s1, s2, m, z):
    result = torch.sub(s1, s2)
    result = result * z/(2*m)
    return result

def error_fn(y_hat, y):
    y = y.numpy()
    y_hat = y_hat.numpy()
    y_hat[0] = np.clip(y_hat[0], 0.000001, 0.999999)
    return np.mean(-y*np.log(y_hat) - (1-y)*np.log(1 - y_hat))

def get_grad(w, b, X, y, m = 0.0001):
    d_w = w.size()[0]
    d_b = b.size()[0]
    z_w = torch.randn(d_w, 1)
    z_b = torch.randn(d_b, 1)
    final_grad_w = torch.zeros(d_w, 1)
    final_grad_b = torch.zeros(d_b, 1)

    n = X.size(0)

    sigmoid = nn.Sigmoid()

    # Weights
    mm1 = torch.matmul( (w+m*z_w).T , X.T)
    s1 = sigmoid(mm1 + b)
    mm2 = torch.matmul( (w-m*z_w).T , X.T)
    s2 = sigmoid(mm2 + b)
    final_grad_w = out(error_fn(s1, y), error_fn(s2, y), m, z_w)
    
    # Bias
    mm1 = torch.matmul(w.T, X.T)
    mm2 = m*z_b
    s1 = sigmoid(mm1 + b + mm2)
    s2 = sigmoid(mm1 + b - mm2)
    temp = out(error_fn(s1, y), error_fn(s2, y), m, z_b)
    final_grad_b = torch.sum(temp, 1)

    return final_grad_w, final_grad_b