###THIS CODE WROTE IN PYCHARM AND RUN GOOD IN PYCHARM
#Import the libraries
import datetime
import pandas as pd
import numpy as np
import json
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import statistics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.svm import SVC

#Knn
#for this classification, I provided the mean and STD f1-score of 1,000 random split train-test for each value, when the value of k is 1 to 20.
# I founded that the high value of k is 5  0.923500, when k=5. So, we need to choose in k=5.
#This valueis good to this data. If we are taking another data, maybe the high value of k is will be not "5".
def Knn_reg (X_train, X_test, y_train, y_test):
    dic_Knn = {}
    for j in range(1, 21):
        classifier = KNeighborsClassifier(n_neighbors=j, metric='minkowski', p=2)
        classifier.fit(X_train, y_train)
        test_pred = classifier.predict(X_test)
        sco = f1_score(y_test, test_pred)
        dic_Knn[j] = sco
    row_KNN.append(dic_Knn)
    df_KNN = pd.DataFrame(row_KNN)
    if numOfCurrent == 999:
        mean = df_KNN.mean()
        std = df_KNN.std()
        df_result = pd.DataFrame()
        df_result["mean"] = mean
        df_result["std"] = std
        print() #space
        print("Knn results:")
        print(df_result)
        return df_result
#Logistic regression
#for this classification, I provided the mean and STD f1-score for 1,000 random train-test split.
def log_reg (X_train, X_test, y_train, y_test):
    LogReg = LogisticRegression(solver='liblinear')
    LogReg.fit(X_train, y_train)
    y_pred = LogReg.predict(X_test)
    sco = f1_score(y_test, y_pred)
    row_log.append(sco)
    if numOfCurrent == 999:
        df_log = pd.DataFrame(row_log)
        mean = df_log.mean()
        std = df_log.std()
        df_result = pd.DataFrame()
        df_result["mean"] = mean
        df_result["std"] = std
        print() #space
        print("Log regression results:")
        print(df_result)
        return df_result
#Linear SVC
#for this classification, I provided the mean and STD f1-score for 1,000 random train-test split.
def lin_reg(X_train, X_test, y_train, y_test):
    linReg = svm.SVC(kernel='linear', C=1)
    linReg.fit(X_train, y_train)
    y_pred = linReg.predict(X_test)
    sco = f1_score(y_test, y_pred)
    row_lin.append(sco)
    if numOfCurrent == 999:
        df_lin = pd.DataFrame(row_lin)
        mean = df_lin.mean()
        std = df_lin.std()
        df_result = pd.DataFrame()
        df_result["mean"] = mean
        df_result["std"] = std
        print() #space
        print("Linear SVC results:")
        print(df_result)
        return df_result

#Polynomial SVC of degree m
#for this classification, I provided the mean and STD f1-score of 1,000 random split train-test for each value with value of m is 2,3,4,5.
#I founded that the high value of f1-score is 0.841065 when m=3.
#like K-NN, this valueis good to this data. If we are taking another data, maybe the high value of m is will be not "3".
def poly_reg(X_train, X_test, y_train, y_test):
    dic_poly = {}
    val_poly = [2,3,4,5]
    for j in val_poly:
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        svclassifier = SVC(kernel='poly', degree=j)
        svclassifier.fit(X_train, y_train)
        y_pred = svclassifier.predict(X_test)
        sco = f1_score(y_test, y_pred)
        dic_poly[j] = sco
    row_poly.append(dic_poly)
    df_poly = pd.DataFrame(row_poly)
    if numOfCurrent == 999:
        mean = df_poly.mean()
        std = df_poly.std()
        df_result = pd.DataFrame()
        df_result["mean"] = mean
        df_result["std"] = std
        print() #space
        print("Polynomial SVC results:")
        print(df_result)
        return df_result

# Gaussian SVC
#For this classification, I founded the mean and STD f1-score of 1,000 random split train-test for each value, when the value of C is:0.2, 0.5, 1.2, 1.8, 3.
#I founded that the high value of f1-score is 0.928642 when m=3.
#like K-NN and polynomual SVC, this valueis good to this data. If we are taking another data, maybe the high value of m is will be not "3".
def gas_reg(X_train, X_test, y_train, y_test):
    dic_gas = {}
    val_gas = [0.2, 0.5, 1.2, 1.8, 3]
    for j in val_gas:
        svc = SVC(kernel='rbf', gamma=j)
        svc.fit(X_train, y_train)
        y_pred = svc.predict(X_test)
        sco = f1_score(y_test, y_pred)
        dic_gas[j] = sco
    row_gas.append(dic_gas)
    df_gas = pd.DataFrame(row_gas)
    if numOfCurrent == 999:
        mean = df_gas.mean()
        std = df_gas.std()
        df_result = pd.DataFrame()
        df_result["mean"] = mean
        df_result["std"] = std
        print() #space
        print("Gaussian SVC results:")
        print(df_result)
        return df_result
#Function that calculate the mean of f1-score value.
def Average(row2):
    return sum(row2) / len(row2)


##Load the dataset
data = r'C:\Einav\dataset.csv'
df = pd.read_csv(data, encoding="ISO-8859-1")
df = pd.DataFrame(df)


##Split the data to x and y
X = df.iloc[:, [0, 1]].values
y = df.iloc[:, [2]].values

#global variable that relevant of all the code.
global row_KNN
row_KNN = []
global row_log
row_log = []
global row_lin
row_lin =[]
global row_poly
row_poly =[]
global row_gas
row_gas =[]

#Spilt random train-test data, chek all models.
#Check mean and STD f1-score of each model and his values for each run (i=1,2......1000)
#Do Standartization
for i in range(1,1000):
    numOfCurrent = i
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=None)
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    knn_reg_result = Knn_reg(X_train, X_test, y_train, y_test)
    log_reg_result = log_reg(X_train, X_test, y_train, y_test)
    lin_reg_result = lin_reg(X_train, X_test, y_train, y_test)
    poly_reg_result = poly_reg(X_train, X_test, y_train, y_test)
    gas_reg_result = gas_reg(X_train, X_test, y_train, y_test)
