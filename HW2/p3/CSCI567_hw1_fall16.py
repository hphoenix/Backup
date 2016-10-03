#! /usr/bin/python
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as sps

# import training, test data.
train = np.loadtxt('train.txt', delimiter=',')
test = np.loadtxt('test.txt', delimiter=',')
##### Naive Bayes #####
# compute conditional distribution parameters for each attribute
# compute mean for each attribute class.
mean = np.zeros(shape=(7, 9), dtype=float)
stdev = np.zeros(shape=(7, 9), dtype=float)
priorClass = np.zeros(shape=(7), dtype=float)
for row in range(len(train)):
    c = int(train[row][10] - 1)  #class -1 due to Zero indexing of mean array
    priorClass[c] = priorClass[c] + 1
    for attr in range(0, 9):
        mean[c][attr] = mean[c][attr] + train[row][attr + 1]  #train[attr+1] due to ignoring first column
for c in range(0, 7):
    priorClass[c] = priorClass[c] / len(train)
    for attr in range(0, 9):
        mean[c][attr] = mean[c][attr] / len(train)
# compute standard deviation by class
for row in range(len(train)):
    c = int(train[row][10] - 1)  # class-1 due to Zero indexing of mean array
    for attr in range(0, 9):
        stdev[c][attr] = stdev[c][attr] + pow((train[row][attr + 1] - mean[c][attr]), 2)
for c in range(0, 7):
    for attr in range(0, 9):
        stdev[c][attr] = stdev[c][attr] / len(train)
        stdev[c][attr] = np.sqrt(stdev[c][attr])
# training accuracy
predClass = np.zeros(shape=(7), dtype=float)


def predict(dataset, row):
    pClass = 0
    for c in range(0, 7):
        predClass[c] = priorClass[c]  # p(y=Ci)
        for attr in range(0, 9):
            if stdev[c][attr] != 0: # To handle cases when stdev is 0
                predClass[c] = predClass[c] * sps.norm(mean[c][attr], stdev[c][attr]).pdf(dataset[row][attr + 1])#p(xi/y=Ci)
            elif (stdev[c][attr] == 0) and (mean[c][attr] != dataset[row][attr + 1]):
                predClass[c] = 0

    if predClass[c] > predClass[pClass]:
        pClass = c
      # print pClass
    return pClass

print "Calculating Naive Bayes model accuracy.."
trainAcc = 0.0
for row in range(len(train)):
    pClass = predict(train, row)
    if pClass == (train[row][10] - 1):  # pclass from 0 to 6 not 1 to 7
        trainAcc = trainAcc + 1
trainAcc = trainAcc * 100 / len(train)
print "Naive Bayes Training Accuracy is:", trainAcc
# testing accuracy
testAcc = 0.0
for row in range(len(test)):
    pClass = predict(test, row)
    if pClass == (test[row][10] - 1):  # pclass from 0 to 6 not 1 to 7
        testAcc = testAcc + 1
testAcc = testAcc * 100 / len(test)
print "Naive Bayes Testing Accuracy is:", testAcc
######### KNN #########
# Mean and N-1 Std dev
def procStd(dataset): #To handle stdev of 0
    for i in range(len(dataset)):
        if dataset[i]==0:
            dataset[i] = 0.1


trainStd = train.std(axis=0) * (len(train)/(len(train)-1))
testStd = test.std(axis=0) * (len(test)/(len(test)-1))
procStd(trainStd)
procStd(testStd)
normTrain = (train - train.mean(axis=0)) / trainStd
normTest = (test - test.mean(axis=0)) / testStd


def knnPredict(trainFlag, k, argArr):  # trainFlag=0 is test, 1 is train
    knnPredClass = np.zeros(shape=(7), dtype=float) # Array to count votes per class
    pClass = np.zeros(shape=7, dtype=float)
    classL = np.zeros(shape=(7), dtype=float)
    start = trainFlag  # Leave one out for training. Ignoring first index in sorted distance array
    for j in range(k): # Using the row index from sorted argArr to get the corresponding class value. Skip first value pointed by argArr if training
        classL[j] = int(train[argArr[start + j]][10] - 1)
        knnPredClass[int(train[argArr[start + j]][10] - 1)] = knnPredClass[int(train[argArr[start + j]][10] - 1)] + 1 #Voting
    # Sort while breaking ties
    pClass = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    for j in range(len(knnPredClass)):
        for i in range(len(knnPredClass) - 1):
            if knnPredClass[i] > knnPredClass[i + 1]:
                temp = knnPredClass[i + 1]
                knnPredClass[i+1] = knnPredClass[i]
                knnPredClass[i] = temp
                temp1 = pClass[i + 1]
                pClass[i + 1] = pClass[i]
                pClass[i] = temp1
            elif knnPredClass[i] == knnPredClass[i + 1]: # Tie breaking
                swap = 0
                for l in range(k):
                    if i == classL[l]:
                        swap = 1
                        break
                    elif (i + 1) == classL[l]:
                        swap = 0
                        break
                if swap == 1:
                    temp = knnPredClass[i + 1]
                    knnPredClass[i + 1] = knnPredClass[i]
                    knnPredClass[i] = temp
                    temp1 = pClass[i + 1]
                    pClass[i + 1] = pClass[i]
                    pClass[i] = temp1
            #elif knnPredClass[i] < knnPredClass[i + 1]:
    return pClass[6] # return class with Maximum votes


def knnAcc(trainFlag, k):
    dataset = normTrain if trainFlag else normTest
    dataset2 = train if trainFlag else test
    acc1 = 0.0
    acc2 = 0.0
    for row in range(len(dataset)):
        pClass = 0.0
        knnL1 = np.zeros(shape=(len(train)), dtype=float)
        knnL2 = np.zeros(shape=(len(train)), dtype=float)
        argL1 = np.zeros(shape=(len(train)), dtype=int)
        argL2 = np.zeros(shape=(len(train)), dtype=int)
        for row1 in range(len(normTrain)):
            for attr in range(1, 10):
                knnL1[row1] = knnL1[row1] + abs(dataset[row][attr] - normTrain[row1][attr])
                knnL2[row1] = knnL2[row1] + pow(dataset[row][attr] - normTrain[row1][attr], 2)
            knnL2[row1] = np.sqrt(knnL2[row1])
        argL1 = np.argsort(knnL1)
        argL2 = np.argsort(knnL2)
        pClass = knnPredict(trainFlag, k, argL1)
        if pClass == (dataset2[row][10] - 1):  # pclass from 0 to 6 not 1 to 7
            acc1 = acc1 + 1
        pClass = knnPredict(trainFlag, k, argL2)
        if pClass == (dataset2[row][10] - 1):  # pclass from 0 to 6 not 1 to 7
            acc2 = acc2 + 1
    acc1 = acc1 * 100 / len(dataset)
    acc2 = acc2 * 100 / len(dataset)
    print "L1 metric is: ", acc1
    print "L2 metric is: ", acc2


for k in range(1,8,2):
    print "KNN Training accuracy for K =", k, "using"
    knnAcc(1, k)
    print "KNN Testing accuracy for K =", k, "using"
    knnAcc(0, k)