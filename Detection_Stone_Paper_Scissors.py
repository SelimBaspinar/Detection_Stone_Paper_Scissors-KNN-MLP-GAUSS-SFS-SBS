# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 19:03:30 2021

@author: SelimPc
"""


import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from time import time
from sklearn.preprocessing import StandardScaler



#İmport Data
data=[]
for file in os.listdir("./input/HandMuscleActivity"):
    read = pd.read_csv("./input/HandMuscleActivity/"+file, header = None,dtype = np.float32)
    data.append(read)
df = pd.concat(data)     


#Data input and labels
targets_numpy = df.iloc[:, -1].values
features_numpy =  df.iloc[:, :-1].values 

# train test split. Size of train data is 70% and size of test data is 30%. 
features_train, features_test, targets_train, targets_test = train_test_split(features_numpy,
                                                                              targets_numpy,
                                                                              test_size = 0.3,
                                                                              random_state = 42) 


#NN MultiLayerPerceptron Classfier
mlp= MLPClassifier(random_state=1, max_iter=400)
y_pred =mlp.fit(features_train, targets_train).predict(features_test)
test_accuracy = mlp.score(features_test, targets_test)
print("MultilayerPerceptron Classifier Acurracy: %%%s"%(int(100*test_accuracy)))
#fscore
print("MultilayerPerceptron F-Score: %s"%(f1_score(targets_test, y_pred, average=None)))
#confusion_matrix
print("MultilayerPerceptron Confusion-Matrix:")
print(confusion_matrix(targets_test, y_pred))

#KNN
knn = KNeighborsClassifier()
y_pred = knn.fit(features_train, targets_train).predict(features_test)
test_accuracy = knn.score(features_test, targets_test)
print("KNN Acurracy: %%%s"%(int(100*test_accuracy)))
#fscore
print("KNN F-Score: %s"%(f1_score(targets_test, y_pred, average=None)))
#confusion_matrix
print("KNN Confusion-Matrix:")
print(confusion_matrix(targets_test, y_pred))

#Gauss
gnb = GaussianNB()
y_pred = gnb.fit(features_train, targets_train).predict(features_test)
test_accuracy = gnb.score(features_test, targets_test)
print("Gauss Acurracy: %%%s"%(int(100*test_accuracy)))
#fscore
print("Gauss F-Score: %s"%(f1_score(targets_test, y_pred, average=None)))
#confusion_matrix
print("Gauss Confusion-Matrix:")
print(confusion_matrix(targets_test, y_pred))

#SFS
tic_fwd = time()
sfs = SequentialFeatureSelector(gnb,scoring='accuracy',direction='forward')
y_pred=sfs.fit(features_train, targets_train)
toc_fwd = time()

#SBS
tic_bwd  = time()
sbs = SequentialFeatureSelector(gnb,scoring='accuracy',direction='backward')
y_pred=sbs.fit(features_train, targets_train)
toc_bwd  = time()
print("SFS")
print(sfs.get_support())
print(sfs.transform(features_train).shape)
print(f"Done in {toc_fwd - tic_fwd:.3f}s")
print("SBS")
print(sbs.get_support())
print(sbs.transform(features_train).shape)
print(f"Done in {toc_bwd - tic_bwd:.3f}s")
#SFS MUCH BETTER THAN SBS



