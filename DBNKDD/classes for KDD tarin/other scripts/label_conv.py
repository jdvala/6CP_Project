#!/usr/bin/env python
import csv

f1 = open('C:/Users/Jay/Desktop/MachineLearning/dataset/NSL-KDD_Processed/NSL_ValidLabels_int.csv', 'rt')
f2 = open('C:/Users/Jay/Desktop/MachineLearning/dataset/NSL-KDD_Processed/NSL_ValidLabels_int3.csv', 'wt')


for line in f1:
    if '0' in line:
     	f2.write('0,0,0,0,0'+'\n')
    
    elif '1' in line:
        f2.write('0,1,0,0,0'+'\n')
        
    elif '2' in line:
        f2.write('0,0,1,0,0'+'\n')
        
    elif '3' in line:
        f2.write('0,0,0,1,0'+'\n')
        
    elif '4' in line:
        f2.write('0,0,0,0,1'+'\n')
    
    else:
        continue
    
f1.close
f2.close
    