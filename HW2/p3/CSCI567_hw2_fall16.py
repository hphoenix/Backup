#! /usr/bin/python
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
import statsmodels.formula.api as smf
boston = load_boston()
### train/test splitting
numAttr = 13
numTrain = 433
numTest = 73
trainAttr = np.empty((numTrain,numAttr))
trainTarg = np.empty((numTrain,))
testAttr = np.empty((numTest,numAttr))
testTarg = np.empty((numTest,))
test = 0
train = 0
for i in range(506):
    if i%7 == 0:
        testAttr[test] = boston.data[i]
        testTarg[test] = boston.target[i]
        test = test + 1
    else:
        trainAttr[train] = boston.data[i]
        trainTarg[train] = boston.target[i]
        train = train + 1
#print test
#print train
### Data analysis
### Histogram
corr = np.zeros(numAttr)
for i in range(numAttr):
    plt.subplot(5, 3, i + 1)
    plt.hist(trainAttr[:,i],bins=10)
    #plt.show() # Clear previous figure
    #plt.savefig('histogram'+ str(i)+'.pdf', format='pdf')
    #plt.clf()
    ret =  sp.stats.pearsonr(trainAttr[:,i], trainTarg[:]) # correlation
    corr[i] = ret[0]
    #print ret[0]
plt.show()
plt.clf()
print "Part 1:"
print "Pearson correlation coefficients for attributees are : \t", corr
### Data Preprocessing. Standardization
### Mean and N-1 Std dev
#def procStd(dataset): #To handle stdev of 0
#    for i in range(len(dataset)):
#        if dataset[i]==0:
#            dataset[i] = 0.1
trainAttrStd = np.std(trainAttr, axis=0) * (numTrain/(numTrain-1))
testAttrStd = np.std(testAttr, axis=0) * (numTest/(numTest-1))
trainTargStd = np.std(trainTarg) * (numTrain/(numTrain-1))
testTargStd = np.std(testTarg) * (numTest/(numTest-1))
#procStd(trainAttrStd)
#procStd(testAttrStd)
#procStd(trainTargStd)
#procStd(testTargStd)
normTrainAttr = (trainAttr - np.mean(trainAttr, axis=0)) / trainAttrStd
normTestAttr = (testAttr - np.mean(testAttr, axis=0)) / testAttrStd
#normTrainTarg = (trainTarg - np.mean(trainTarg)) / trainTargStd
normTrainTarg = trainTarg
normTestTarg = testTarg
#normTestTarg = (testTarg - np.mean(testTarg)) / testTargStd
### Linear Regression
print "# Part 2: Linear Regression #"
#normTrain = np.column_stack((normTrainTarg, normTrainAttr))
#lr = smf.ols(formula='MEDV ~ CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO + B + LSTAT', data=normTrain).fit()
#print lr.params
const = np.ones(numTrain)
X = np.column_stack((const, normTrainAttr))
XT = np.transpose(X)
const = np.ones(numTest)
Xtest = np.column_stack((const, normTestAttr))
XtestT = np.transpose(Xtest)
#print predictTrain
#print predictTest
#print wLMS
#print normTestTarg.shape
### MSE
def MSELR(true, predict):
    se = ((true - predict) ** 2)
    mse = np.mean(se, axis=0)
    return mse
# wLMS = (XT X)^(-1) * XT * y
wLMS = np.dot(np.dot(np.linalg.pinv(np.dot(XT,X)),XT),normTrainTarg)
wLMST = np.transpose(wLMS)
predictTrain = np.dot(wLMST,XT)
predictTest = np.dot(wLMST,XtestT)
print 'MSE for Linear Regression on Training dataset is:\t', MSELR(normTrainTarg, predictTrain)
print 'MSE for Linear Regression on Test dataset is:\t', MSELR(normTestTarg, predictTest)
### Check using Scikit learn
#print "SciKit return values"
#X = normTrainAttr
#y = normTrainTarg
#from sklearn.linear_model import LinearRegression
#lm = LinearRegression()
#lm.fit(X,y)
#print lm.intercept_
#print lm.coef_
#predictSk = lm.predict(normTestAttr)
#print MSELR(normTestTarg, predictSk)
### Ridge Regression
print "# Part 3: Ridge Regression #"
lam = [0.01, 0.1, 1.0, 10]
for i in range(len(lam)):
    wLMS = np.dot(np.dot(np.linalg.pinv(np.dot(XT,X) + np.dot(lam[i], np.identity(numAttr+1))),XT),normTrainTarg)
    wLMST = np.transpose(wLMS)
    predictTrain = np.dot(wLMST, XT)
    predictTest = np.dot(wLMST, XtestT)
    print 'MSE for Ridge Regression on Training dataset with lambda', lam[i],' is:\t', MSELR(normTrainTarg, predictTrain)
    print 'MSE for Ridge Regression on Test dataset with lambda', lam[i],' is:\t', MSELR(normTestTarg, predictTest)
### Ridge regression, Hyper parameter tuning using Cross Validation
# Split
indices = np.random.permutation(X.shape[0])
k = 10 # 10-fold CV
#Split: 7 x 43 , 3 x 44
CVIndex1 = np.empty((7,43), dtype='int')
CVIndex2 = np.empty((3,44), dtype='int')
i1 = 0
for i in range(k):
    #print i
    if i<7:
        CVIndex1[i] = indices[i1 : i1+43]
        i1 = i1+43
        #print CVIndex1[i]
    else:
        CVIndex2[i-7] = indices[i1 : i1+44]
        i1 = i1+44
        #print CVIndex2[i-7]
CVTrainIndex = []
MSETrain = np.zeros(10)
MSETest = np.zeros(10)
numSteps = 10
avgMSETest = np.zeros(numSteps)
lambd = np.linspace(0.0001, 10, numSteps)
for i in range(numSteps):
    lam = lambd[i]
    for l in range(k):
        if l<7:
            CVTestIndex = CVIndex1[l]
        else:
            CVTestIndex = CVIndex2[l-7]
        for j in range(k):
            if j != l:
                if j<7:
                    for m in range(len(CVIndex1[j])):
                        #print CVIndex1[j][m]
                        CVTrainIndex.append(CVIndex1[j][m])
                else:
                    for m in range(len(CVIndex2[j-7])):
                        #print CVIndex2[j-7][m]
                        CVTrainIndex.append(CVIndex2[j-7][m])
        CVTrain, CVTest = X[CVTrainIndex,:], X[CVTestIndex,:] # Attributes
        CVTrainTarg, CVTestTarg = normTrainTarg[CVTrainIndex], normTrainTarg[CVTestIndex] # Targets
        CVTrainT = np.transpose(CVTrain)
        CVTestT = np.transpose(CVTest)
        wLMS = np.dot(np.dot(np.linalg.pinv(np.dot(CVTrainT, CVTrain) + np.dot(lam, np.identity(numAttr + 1))), CVTrainT),
                      CVTrainTarg)
        wLMST = np.transpose(wLMS)
        predictTrain = np.dot(wLMST, CVTrainT)
        predictTest = np.dot(wLMST, CVTestT)
        MSETrain[l] = MSELR(CVTrainTarg, predictTrain)
        MSETest[l] = MSELR(CVTestTarg, predictTest)
    avgMSETest[i] = np.mean(MSETest)
    #print 'Average MSE for Ridge Regression on Training dataset with lambda', lam, ' is:\t', np.mean(MSETrain)
    print 'Average MSE for Ridge Regression on Cross Validation Test dataset with lambda', lam, ' is:\t', avgMSETest[i]
#print lambd
#print avgMSETest
plt.plot(lambd, avgMSETest)
#plt.show()
plt.savefig('lambda vs avgMSETest.pdf', format='pdf')
plt.clf()
### Feature Selection
### Correlation
#print np.abs(corr)
plt.plot(np.abs(corr))
plt.savefig('Absolute Correlation of features.pdf', format='pdf')
plt.clf()
print "# part 3.3(a) #"
print "The features with highest absolute correlation in order are: LSTAT, RM , PTRATIO, INDUS"

def LR(X, Xtest, TrainTarg):
    XT = np.transpose(X)
    XtestT = np.transpose(Xtest)
    wLMS = np.dot(np.dot(np.linalg.pinv(np.dot(XT,X)),XT),TrainTarg)
    wLMST = np.transpose(wLMS)
    predictTrain = np.dot(wLMST,XT)
    predictTest = np.dot(wLMST,XtestT)
    return predictTrain, predictTest
trainAttr3a = np.column_stack((X[:,0],X[:,13],X[:,6],X[:,11],X[:,3]))
#print trainAttr3a
testAttr3a = np.column_stack((Xtest[:,0],Xtest[:,13],Xtest[:,6],Xtest[:,11],Xtest[:,3]))
predictTrain,predictTest = LR(trainAttr3a, testAttr3a, normTrainTarg)
print 'MSE for Linear Regression on Training dataset is:\t', MSELR(normTrainTarg, predictTrain)
print 'MSE for Linear Regression on Test dataset is:\t', MSELR(normTestTarg, predictTest)
# recursive correlation
print "# part 3.3(b) #"
features = {1:'CRIM',2:'ZEN',3:'INDUS',4:'CHAS',5:'NOX',6:'RM',7:'AGE',8:'DIS',9:'RAD',10:'TAX',11:'PTRATIO',12:'B',13:'LSTAT'}
residue = normTrainTarg
trainAttr3b = np.column_stack((X[:,0],X[:,13]))
testAttr3b = np.column_stack((Xtest[:,0],Xtest[:,13]))
print 'Feature is:\t',features[13]
# corr
for l in range(4):
    predictTrain, predictTest = LR(trainAttr3b, testAttr3b, normTrainTarg)
    residue = normTrainTarg - predictTrain
    for i in range(numAttr):
        ret = sp.stats.pearsonr(normTrainAttr[:, i], residue[:])
        corr[i] = np.abs(ret[0])
    max = np.argmax(corr)
    trainAttr3b = np.column_stack((trainAttr3b,X[:,max+1]))
    testAttr3b = np.column_stack((testAttr3b,Xtest[:,max+1]))
    if l==3:
        print 'MSE for Linear Regression on Training dataset is:\t', MSELR(normTrainTarg, predictTrain)
        print 'MSE for Linear Regression on Test dataset is:\t', MSELR(normTestTarg, predictTest)
    else:
        print 'Feature is:\t', features[max + 1]
#Brute force
print "# part 3.3(c) #"
count = 0
best = np.zeros(4, dtype='int')
bestMSETrain = 1000.0
for i in range(1,numAttr+1):
    for j in range(i+1,numAttr+1):
        for k in range(j+1,numAttr+1):
            for l in range(k+1,numAttr+1):
                trainAttr3c = np.column_stack((X[:, 0], X[:, i], X[:, j], X[:, k], X[:, l]))
                testAttr3c = np.column_stack((Xtest[:, 0], Xtest[:, i], Xtest[:, j], Xtest[:, k], Xtest[:, l]))
                predictTrain, predictTest = LR(trainAttr3c, testAttr3c, normTrainTarg)
                MSETrain = MSELR(normTrainTarg, predictTrain)
                #MSETest = MSELR(normTestTarg, predictTest)
                if MSETrain < bestMSETrain:
                    bestMSETrain = MSETrain
                    best = ([i,j,k,l])
trainAttr3c = np.column_stack((X[:, 0], X[:, best[0]], X[:, best[1]], X[:, best[2]], X[:, best[3]]))
testAttr3c = np.column_stack((Xtest[:, 0], Xtest[:, best[0]], Xtest[:, best[1]], Xtest[:, best[2]], Xtest[:, best[3]]))
predictTrain, predictTest = LR(trainAttr3c, testAttr3c, normTrainTarg)
print 'The four features after brute force aproach are:', features[best[0]], features[best[1]], features[best[2]], features[best[3]]
print 'MSE for Linear Regression on Training dataset is:\t', bestMSETrain
#print 'MSE for Linear Regression on Training dataset is:\t', MSELR(normTrainTarg, predictTrain)
print 'MSE for Linear Regression on Test dataset is:\t', MSELR(normTestTarg, predictTest)
# Polynomial Feature Expansion
print "# Part 3.4 #"
for i in range(0, numAttr):
    for j in range(i, numAttr):
        new = np.multiply(normTrainAttr[:,i],normTrainAttr[:,j])#scalar multiplication to get a new feature
        newStd = np.std(new, axis=0) * (numTrain / (numTrain - 1))
        if newStd == 0.0:
            newStd = 0.1
        newNorm = (new - np.mean(new, axis=0)) / newStd
        X = np.column_stack((X,newNorm))
for i in range(0, numAttr):
    for j in range(i, numAttr):
        new = np.multiply(normTestAttr[:,i],normTestAttr[:,j])#scalar multiplication to get a new feature
        newStd = np.std(new, axis=0) * (numTest / (numTest - 1))
        if newStd == 0.0:
            newStd = 0.1
        newNorm = (new - np.mean(new, axis=0)) / newStd
        Xtest = np.column_stack((Xtest,newNorm))
#print Xtest.shape
predictTrain,predictTest = LR(X, Xtest, normTrainTarg)
print 'MSE for Linear Regression on Training dataset is:\t', MSELR(normTrainTarg, predictTrain)
print 'MSE for Linear Regression on Test dataset is:\t', MSELR(normTestTarg, predictTest)