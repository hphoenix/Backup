#! /usr/bin/python
import re
import sys
import math
inFile = open("p3.txt",'r')
x = []
y = []
i = 0
for line in inFile:
    m = re.match(r"(\-?\d+)\s+(\-?\d+)",line)
    x.append(int(m.group(1)))
    y.append(int(m.group(2)))
    i = i+1
N = i
# Mean
x_mean = 0
y_mean = 0
for i in range(len(x)):
    x_mean = x_mean+x[i]
    y_mean = y_mean+y[i]
x_mean = float(x_mean)/float(N)
y_mean = float(y_mean)/float(N)
# N-1 Standard deviation
x_sd = 0
y_sd = 0
for i in range(len(x)):
    x_sd = x_sd + pow((x[i]-x_mean),2)
    y_sd = y_sd + pow((y[i]-y_mean),2)
x_sd = float(x_sd)/(N-1)
x_sd = math.sqrt(x_sd)
y_sd = float(y_sd)/(N-1)
y_sd = math.sqrt(y_sd)
print ("x_mean:",x_mean," y_mean:",y_mean," x_sd:",x_sd," y_sd:",y_sd)
a = float(20 - x_mean)/x_sd
b = float(7 - y_mean)/y_sd
print "norm. target point"
print (a,b)
#Normalize
print "norm. inputs"
for i in range(len(x)):
   x[i] =  float(x[i] - x_mean)/x_sd
   y[i] =  float(y[i] - y_mean)/y_sd
   print (x[i],y[i])
#L1
L1 = []
print "L1 values"
for i in range(len(x)):
    L1.append(abs(a-x[i])+abs(b-y[i]))
    print (i+1, L1[i])
#L2
L2 = []
print "L2 values"
for i in range(len(x)):
    L2.append(math.sqrt(pow(a-x[i],2)+pow(b-y[i],2)))
    print (i+1, L2[i])


inFile.close()
