
#for 2 classes Normal(0) vs DoS(1) vs Attack(2)
dataset1='C:/Users/Jay/Desktop/class2.csv'


f1 = open('C:/Users/Jay/Desktop/MachineLearning/dataset/NSL-KDD_Processed/NSL_TrainLabels_int.csv', 'rt')
f2 = open(dataset1, 'wt')

for line in f1:
    if '0' in line:
        f2.write('0'+'\n')
    elif '1' in line:
        f2.write('1'+'\n')
    else:
        f2.write('2'+'\n')

f1.close()
f2.close()