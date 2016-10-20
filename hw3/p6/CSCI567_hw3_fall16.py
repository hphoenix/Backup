#! /usr/bin/python
import numpy as np
#import scipy as sp
from scipy import io
import pandas as pd
import matplotlib.pyplot as plt
from svmutil import *
### BIAS-VARIANCE ###
x = np.random.uniform(-1,1,10)










### SVM ###
train = io.loadmat("phishing-train.mat")
test = io.loadmat("phishing-test.mat")
print train.viewkeys()
trainLabels = train['label']
trainFeaturesOld = train['features']
testLabels = test['label']
testFeaturesOld = test['features']
#print trainFeaturesOld.shape
#trainFeatures = np.ndarray((2000, 46))
#testFeatures = np.ndarray((2000,46))
trainFeatures = []
testFeatures = []
# Pre-process. Why in this way?
def preprocess(old, new):
    for i in range(2000):
        index = 0
        sample = []
        for j in range(30):
            if j == 1 or j == 6 or j == 7 or j == 13 or j == 14 or j == 15 or j == 25 or j == 28:
                if old[i][j] == -1:
                    sample.append(1)
                    index = index + 1
                    sample.append(0)
                    index = index + 1
                    sample.append(0)
                elif old[i][j] == 0:
                    sample.append(0)
                    index = index + 1
                    sample.append(1)
                    index = index + 1
                    sample.append(0)
                elif old[i][j] == 1:
                    sample.append(0)
                    index = index + 1
                    sample.append(0)
                    index = index + 1
                    sample.append(1)
            else:
                if old[i][j] == -1:
                    sample.append(0)
                else:
                    sample.append(old[i][j])
            index = index + 1
        new.append(sample)
preprocess(trainFeaturesOld, trainFeatures)
preprocess(testFeaturesOld, testFeatures)
#print features.shape #2000x30
#print test
#print trainFeaturesOld[:,27]
#print trainFeatures[42]
#print trainFeatures[43]
#print trainFeatures[44]
#print labels.shape
### LIBSVM ###
print trainFeaturesOld
print trainFeatures
trainLabels = np.transpose(trainLabels)
testLabels = np.transpose(testLabels)
#print trainLabels.shape
Prob = svm_problem(trainLabels,trainFeatures)
cvArray = [pow(4,-6),pow(4,-5),pow(4,-4),pow(4,-3),pow(4,-2),pow(4,-1),pow(4,0),pow(4,1),pow(4,2)]
print cvArray
def findMax(inp):
    index = 0
    max = 0.0
    for i in range(len(inp)):
        if inp[i] > max:
            max = inp[i]
            index = i
    return index
linearAcc = []
for c in cvArray:
    s = '-t 0 -v 3 -c '+ `c`
    #print s
    linearAcc.append(svm_train(Prob,s))
### Polynomial kernel
cvArray = [pow(4,-3),pow(4,-2),pow(4,-1),pow(4,0),pow(4,1),pow(4,2),pow(4,3),pow(4,4),pow(4,5),pow(4,6),pow(4,7)]
degreeArray = [1,2,3]
polyAcc = np.zeros((3,11))
ci = 0
for c in cvArray:
    di = 0
    for d in degreeArray:
        s = '-t 1 -v 3 -c '+ `c` +' -d ' + `d`
        polyAcc[di][ci] = svm_train(Prob,s)
        di = di+1
    ci = ci+1
### RBF kernel
cvArray = [pow(4,-3),pow(4,-2),pow(4,-1),pow(4,0),pow(4,1),pow(4,2),pow(4,3),pow(4,4),pow(4,5),pow(4,6),pow(4,7)]
gammaArray = [pow(4,-7),pow(4,-6),pow(4,-5),pow(4,-4),pow(4,-3),pow(4,-2),pow(4,-1)]
rbfAcc = np.zeros((7,11))
ci = 0
for c in cvArray:
    di = 0
    for d in gammaArray:
        s = '-t 2 -v 3 -c '+ `c` +' -g ' + `d`
        rbfAcc[di][ci] = svm_train(Prob,s)
        di = di+1
    ci = ci+1
### Report
print "# Linear SVM"
print findMax(linearAcc)
print linearAcc[findMax(linearAcc)]
print "# Polynomial SVM"
for i in range(len(degreeArray)):
    print findMax(polyAcc[i])
    print polyAcc[i][findMax(polyAcc[i])]
print "# RBF SVM"
for i in range(len(gammaArray)):
    print findMax(rbfAcc[i])
    print rbfAcc[i][findMax(rbfAcc[i])]
# Gamma 4^-1, c = pow(4,1)
# Test
gamma = pow(4,-1)
c = pow(4,1)
s = '-t 2 -c ' + `c` + ' -g ' + `gamma`
m = svm_train(Prob, s)
#p_l = testLabels
p_labels, p_acc, p_vals = svm_predict(testLabels,testFeatures,m)
ACC, MSE, SCC = evaluations(testLabels,p_labels)
print "The accuracy on test set is: " + ACC