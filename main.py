import os
import cv2
import pickle
import numpy as np
import pdb
import requests
from collections import defaultdict
import random 
import time
import sklearn

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import *

from functools import wraps
from time import time as _timenow 
from sys import stderr

from sklearn import decomposition, discriminant_analysis, linear_model, svm, tree
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
    
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

from scipy.misc import imread
from PIL import Image
from sklearn.neural_network import MLPClassifier
from functools import wraps
from sklearn import tree
from time import time as _timenow 
from sys import stderr
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import decomposition, discriminant_analysis, linear_model, svm, tree
from sklearn.metrics import f1_score, accuracy_score 
import sys 

dim = sys.argv[1]
cla_ss = sys.argv[2]

def  load_cifarload_ci ():
    trn_data, trn_labels, tst_data, tst_labels = [], [], [], []
    def unpickle(file):
        with open(file, 'rb') as fo:
            data = pickle.load(fo, encoding='latin1')
        return data
    for i in trange(1):
        batchName = './data/data_batch_{0}'.format(i + 1)
        unpickled = unpickle(batchName)
        trn_data.extend(unpickled['data'])
        trn_labels.extend(unpickled['labels'])
    unpickled = unpickle('./data/test_batch')
    tst_data.extend(unpickled['data'])
    tst_labels.extend(unpickled['labels'])
    return trn_data, trn_labels, tst_data, tst_labels

def image_prep(image):
	num = len(image)
	siz = len(image[0])
	x_mat = np.zeros((num,siz))
	for i in range(num):
		x_mat[i] = (image[i])
	mn = np.mean(x_mat,axis=0)
	std_dev = np.std(x_mat,axis=0)
	x_mat = (x_mat - mn)/std_dev
	return x_mat

def  evaluate(target, predicted):
    f1 = f1_score(target, predicted, average='micro')
    acc = accuracy_score(target, predicted)
    return f1, acc

def reduce_dim(data,**kwargs):
    ''' performs dimensionality reduction'''
    if kwargs['method'] == 'nan':
        return data
    if kwargs['method'] == 'pca':
        dim_red = decomposition.PCA(n_components=0.87,svd_solver ='full')
        dim_red.fit(data)
        return dim_red.transform(data)
    if kwargs['method'] == 'lda':
        dim_red = discriminant_analysis.LinearDiscriminantAnalysis(n_components=9)
        dim_red.fit(data, kwargs['label'])
        return dim_red.transform(data)

def classify(X, y, **kwargs):
    ''' trains a classifier by taking input features
        and their respective targets and returns the trained model'''
    if kwargs['method'] == 'SVM_rbf':
        clf = svm.SVC(kernel = 'rbf',C=kwargs['C'], decision_function_shape='ovo')
        clf.fit(X,y)
    if kwargs['method'] == 'SVM_linear_soft':
        clf = svm.SVC(kernel='linear',C=kwargs['C'])
        clf.fit(X,y)
    if kwargs['method'] == 'log_reg':
        clf = linear_model.LogisticRegression(C=kwargs['C'], solver='lbfgs', multi_class='multinomial')
        clf.fit(X,y)
    if kwargs['method'] == 'mlp':
        clf = sklearn.neural_network.MLPClassifier(activation = kwargs['activation'],alpha = kwargs['alpha'],learning_rate_init= kwargs['learning_rate_init'])
        clf.fit(X,y)
    if kwargs['method'] == 'decision_tree':
        clf = tree.DecisionTreeClassifier(max_depth = kwargs['max_depth'])
        clf.fit(X,y)
    return clf

def test(clf,tst_dt):
    '''takes test data and trained classifier model,
    performs classification and prints accuracy and f1-score'''
    pred_lb = clf.predict(tst_dt)
    return pred_lb

def main():
    trn_dt1 , trn_lb , tst_dt1 ,tst_lb = load_cifarload_ci()
    trn_dt = np.array(image_prep(trn_dt1))
    tst_dt = np.array(image_prep(tst_dt1))
    ''' perform dimesioality reduction/feature extraction and classify the features into one of 10 classses
        print accuracy and f1-score.
        '''
    print(trn_dt.shape)
    trn_dt = reduce_dim(trn_dt,method = dim,label = trn_lb)
    tst_dt = reduce_dim(tst_dt,method = dim,label = tst_lb)
    trn_dt,trn_dt_cr, trn_lb, trn_lb_cr = train_test_split(trn_dt, trn_lb,test_size = 0.20) 
    clsf = classify(trn_dt,trn_lb,method = cla_ss,C=1,activation = 'relu',max_depth = 10,alpha = 0.0001,learning_rate_init = 0.001)
    pred_lb = test(clsf,tst_dt)
    f_score, accuracy_ = evaluate(tst_lb,pred_lb)
    print('Val - F1 score: {}\n Accuracy: {}'.format(f_score, accuracy_))

if __name__ == '__main__':
    main()
   

