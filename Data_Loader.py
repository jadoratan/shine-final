from PIL import Image
import os
import numpy as np
import torch
import random
import SGD2

path = fr"C:\Users\redel\Desktop\Code\SHINE\Project\brain_tumor_dataset"
cwd = os.chdir(path)
dir_list = os.listdir(cwd) 

no_path = path + "\\" + dir_list[0]
yes_path = path + "\\" + dir_list[1]
no_list = os.listdir(no_path)
yes_list = os.listdir(yes_path)

dataset_size = len(no_list) + len(yes_list)
X = []
y = [] # 0-no, 1-yes

total = 0
count_end_3 = 0

for file in no_list:
    file = no_path + "\\" + fr"{file}"
    img = Image.open(file)
    img = img.resize((256, 256))
    numpy_array = np.array(img)

    if ( (numpy_array.ndim==3) and (numpy_array.shape[2]==3) ):
        count_end_3 += 1
        X.append(numpy_array.flatten()) # type: ignore
        y.append(0)

    total += 1
    if (total%10==0):
        print(f"Processed {total} images")


for file in yes_list:
    file = yes_path + "\\" + fr"{file}"
    img = Image.open(file)
    img = img.resize((256, 256))
    numpy_array = np.array(img)

    if ( (numpy_array.ndim==3) and (numpy_array.shape[2]==3) ):
        count_end_3 += 1
        X.append(numpy_array.flatten()) 
        y.append(1)

    total += 1
    if (total%10==0):
        print(f"Processed {total} images")

X = np.array(X)
d = X.shape[1]

for i in range(dataset_size):
    rand_index = random.randint(0,count_end_3-1)
    temp_x = X[0, :]
    temp_y = y[0]
    X = np.delete(X, 0, axis=0)
    y = np.delete(y, 0)
    X = np.insert(X, rand_index, temp_x, axis=0)
    y = np.insert(y, rand_index, temp_y)

# initial values
X = torch.from_numpy(X).to(torch.float)/255
y = torch.from_numpy(y)
w = torch.rand(X.size()[1],1)/d
b = torch.rand(1,1)/d

t = 10000
lr = 0.05

for i in range(t):
    grad_w, grad_b = SGD2.get_grad(w, b, X, y)

    w -= lr * grad_w
    b -= lr * grad_b
    
    if (i%100==0):
        print(f"{i+1} iterations done")


print(SGD2.accuracy(w, b, X, y))