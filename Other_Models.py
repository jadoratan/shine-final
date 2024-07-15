from PIL import Image
import os
import numpy as np
import torch
import random
import SGD2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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



x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# all parameters not specified are set to their defaults
clf = LogisticRegression()
clf.fit(x_train, y_train)
predictions = clf.predict(x_test)
# Use score method to get accuracy of model
score = clf.score(x_test, y_test)
print(f"Logistic Regression: {score}")


clf = svm.SVC()
clf.fit(x_train, y_train)
clf.predict(x_test) # could store predictions as a variable
score = clf.score(x_test, y_test)
print(f"SVM: {score}")


clf = tree.DecisionTreeClassifier()
clf.fit(x_train, y_train)
clf.predict(x_test) # could store predictions as a variable
score = clf.score(x_test, y_test)
print(f"Decision Tree: {score}")


clf = RandomForestClassifier(n_estimators=1)
clf.fit(x_train, y_train)
clf.predict(x_test)
score = clf.score(x_test, y_test)
print(f"Random Forest: {score}")

clf = GaussianNB()
clf.fit(x_train, y_train)
clf.predict(x_test)
score = clf.score(x_test, y_test)
print(f"Gaussian NB: {score}")

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    max_depth=1, random_state=0)
clf.fit(x_train, y_train)
clf.predict(x_test)
score = clf.score(x_test, y_test)
print(f"Gradient Boosting Trees: {score}")

# clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
#     max_depth=1, random_state=0).fit(x_train, y_train)
# clf.score(x_test, y_test)