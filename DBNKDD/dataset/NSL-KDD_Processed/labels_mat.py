#!/usr/bin/env python
#for 2 classes Normal vs Attack
dataset1='C:/Users/Jay/Desktop/MachineLearning/dataset/NSL-KDD_Processed/class1.csv'
#for 3 classes Normal vs Dos Vs other Attack
dataset2='C:/Users/Jay/Desktop/MachineLearning/dataset/NSL-KDD_Processed/class2.csv'
#for 4 class Normal vs Dos vs Probe vs other attack
dataset3='C:/Users/Jay/Desktop/MachineLearning/dataset/NSL-KDD_Processed/class3.csv'
#for 5 classes Normal vs Dos vs Prob vs U2R vs R2I
dataset4='C:/Users/Jay/Desktop/MachineLearning/dataset/NSL-KDD_Processed/class4.csv'


f1 = open('C:/Users/Jay/Desktop/MachineLearning/dataset/NSL-KDD_Processed/NSL_ValidLabels_int.csv', 'rt')
f2 = open(dataset1, 'wt')
f3 = open(dataset2, 'wt')
f4 = open(dataset3, 'wt')
f5 = open(dataset4, 'wt')

# 0 = Normal
# 1 = Dos
# 2 = Probe
# 3 = U2R
# 4 = R2I

for line in f1:
    if '0' in line: #Normal
        #for Normal Vs Attack
        f2.write('1,0,0,0,0'+'\n')
        #for Normal Vs Dos vs Other Attacks
        f3.write('1,0,0,0,0'+'\n')
        #for Normal Vs Dos vs Probe vs Other Attacks
        f4.write('1,0,0,0,0'+'\n')
        #for Normal vs Dos vs Probe vs U2R vs R2I
        f5.write('1,0,0,0,0'+'\n')

    elif '1' in line: #DOS
        #for Normal Vs Attack
        f2.write('0,1,0,0,0'+'\n')
        #for Normal Vs Dos vs Other Attacks
        f3.write('0,1,0,0,0'+'\n')
        #for Normal Vs Dos vs Probe vs Other Attacks
        f4.write('0,1,0,0,0'+'\n')
        #for Normal vs Dos vs Probe vs U2R vs R2I
        f5.write('0,1,0,0,0'+'\n')
        
    elif '2' in line: #Probe
        #for Normal Vs Attack
        f2.write('0,1,0,0,0'+'\n')
        #for Normal Vs Dos vs Other Attacks
        f3.write('0,0,1,0,0'+'\n')
        #for Normal Vs Dos vs Probe vs Other Attacks
        f4.write('0,0,1,0,0'+'\n')
        #for Normal vs Dos vs Probe vs U2R vs R2I
        f5.write('0,0,1,0,0'+'\n')
        
    elif '3' in line: # U2R
        #for Normal Vs Attack
        f2.write('0,1,0,0,0'+'\n')
        #for Normal Vs Dos vs Other Attacks
        f3.write('0,0,1,0,0'+'\n')
        #for Normal Vs Dos vs Probe vs Other Attacks
        f4.write('0,0,0,1,0'+'\n')
        #for Normal vs Dos vs Probe vs U2R vs R2I
        f5.write('0,0,0,1,0'+'\n')
        
    elif '4' in line: #R2I
        #for Normal Vs Attack
        f2.write('0,1,0,0,0'+'\n')
        #for Normal Vs Dos vs Other Attacks
        f3.write('0,0,1,0,0'+'\n')
        #for Normal Vs Dos vs Probe vs Other Attacks
        f4.write('0,0,0,1,0'+'\n')
        #for Normal vs Dos vs Probe vs U2R vs R2I
        f5.write('0,0,0,0,1'+'\n')
        
    else:
        continue
    
f1.close
f2.close