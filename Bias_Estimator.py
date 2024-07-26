import torch
import torch.nn as nn

# weights, bias, transpose of data, 
# mu (used for localizing), 
# z (sample of gaussian distribution, must stay the same across the whole dataset, 
# can be different for weights/biases though) 

def get_grad_b(w, b, X, m = 0.01):
    d = b.size()[0]
    z = torch.randn(d, 1)
    final_grad_b = torch.zeros(d, 1)

    n, x, y = X.size()

    for i in range(n):
        t = torch.transpose(X[i,:,:], 0, 1)
        sigmoid = nn.Sigmoid()

        mm1 = torch.matmul(w, t)
        mm2 = torch.mul(m, z)
        s1 = sigmoid(mm1 + b + mm2)
        s2 = sigmoid(mm1 + b - mm2)

        out = torch.sub(s1, s2)
        out /= (2*m)
        out = torch.matmul(out, z)
        
        final_grad_b += out

    return final_grad_b