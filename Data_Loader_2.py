from PIL import Image
import os
import numpy as np
import torch
import random
import SGD2

# cwd = os.getcwd()
path = fr"C:\Users\redel\Desktop\Code\SHINE\Project\eye_dataset"
cwd = os.chdir(path)
# print("Current working directory:", cwd) 

dir_list = os.listdir(cwd) 
# print("Files and directories in '", cwd, "' :") 
# print(dir_list) 

cataract_path = path + "\\" + dir_list[0]
# diabetic_path = path + "\\" + dir_list[1]
# glaucoma_path = path + "\\" + dir_list[2]
normal_path = path + "\\" + dir_list[3]

cataract_list = os.listdir(cataract_path)
# diabetic_list = os.listdir(diabetic_path)
# glaucoma_list = os.listdir(glaucoma_path)
normal_list = os.listdir(normal_path)

X = []
y = [] # 0 - normal, 1 - cataract
# print(dataset_size)

total = 0

for file in cataract_list:
    file = cataract_path + "\\" + fr"{file}"
    img = Image.open(file)
    img = img.resize((256, 256))
    numpy_array = np.array(img)
    X.append(numpy_array.flatten()) # type: ignore
    y.append(1) # yes, cataract
    total += 1
    if (total%10==0):
        print(f"Processed {total} images")

for file in normal_list:
    file = normal_path + "\\" + fr"{file}"
    img = Image.open(file)
    img = img.resize((256, 256))
    numpy_array = np.array(img)
    X.append(numpy_array.flatten()) # type: ignore
    y.append(0) # no, normal
    total += 1
    if (total%10==0):
        print(f"Processed {total} images")


print(f"Processed {total} images")

X = np.array(X)
d = X.shape[1]
# print(d)
# print("X before:", X.shape)
# print(X[70:75, 70:75])


for i in range(100):
    rand_index = random.randint(0,total-1)
    temp_x = X[0, :]
    temp_y = y[0]
    X = np.delete(X, 0, axis=0)
    y = np.delete(y, 0)
    X = np.insert(X, rand_index, temp_x, axis=0)
    y = np.insert(y, rand_index, temp_y)

    # if (i%20==0  or i==count_end_3-1):
    #     print(y)

print("shuffled")
    
# print("Ends in 3:", count_end_3)
# print("Total:", total)
# print("X after:", X.shape)
# print(X[70:75, 70:75])

# initial values
X = torch.from_numpy(X).to(torch.float)/255
y = torch.from_numpy(y)
w = torch.rand(X.size()[1],1)/d
b = torch.rand(1,1)/d

# print("X size:", X.size())
# print("w size:", w.size())
# print("b size:", b.size())
# print("y size:", y.size())

t = 10000
lr = 0.0001

for i in range(t):
    grad_w, grad_b = SGD2.get_grad(w, b, X, y)

    # print("grad_w.size():",grad_w.size())
    # print("grad_b.size():",grad_b.size())

    w -= lr * grad_w
    b -= lr * grad_b
    
    if (i%100==0):
        print(f"{i+1} iterations done")


print(SGD2.accuracy(w, b, X, y))