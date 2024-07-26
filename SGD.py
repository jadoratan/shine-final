import torch
import torch.nn as nn
import math

def accuracy(w, b, X, y):
    sigmoid = nn.Sigmoid()
    count = 0
    for i in range(X.size(0)):
        val = sigmoid(torch.matmul(w.T, X[i]) + b)
        pred = 0
        if (val >= 0.5):
            pred = 1
    
        if (y[i]==pred):
            count += 1
        
    return count/X.size(0)

def out(s1, s2, m, z):
    result = torch.sub(s1, s2)
    result /= (2*m)
    result = result.item() * z
    return result

def error_fn(y_hat, y):
    if (y_hat == 1):
        y_hat = 0.999999999
    if (y_hat == 0):
        y_hat = 0.000000001
    return -y*math.log(y_hat) - (1-y)*math.log(1 - y_hat)

def get_grad(w, b, X, y, m = 0.01):
    d_w = w.size()[0]
    d_b = b.size()[0]
    z_w = torch.randn(d_w, 1)
    z_b = torch.randn(d_b, 1)
    final_grad_w = torch.zeros(d_w, 1)
    final_grad_b = torch.zeros(d_b, 1)

    n = X.size(0)

    sigmoid = nn.Sigmoid()

    for i in range(n):
        sigmoid = nn.Sigmoid()

        # Weights
        mm1 = torch.matmul( (w+m*z_w).T , torch.unsqueeze(X[i],1))
        s1 = sigmoid(mm1 + b)
        mm2 = torch.matmul( (w-m*z_w).T , torch.unsqueeze(X[i],1))
        s2 = sigmoid(mm2 + b)
        final_grad_w += out(error_fn(s1, y[i]), error_fn(s2, y[i]), m, z_w)

        # Bias
        mm1 = torch.matmul(w.T, torch.unsqueeze(X[i],1))
        mm2 = m*z_b
        s1 = sigmoid(mm1 + b + mm2)
        s2 = sigmoid(mm1 + b - mm2)
        final_grad_b += out(error_fn(s1, y[i]), error_fn(s2, y[i]), m, z_b)

    return final_grad_w, final_grad_b