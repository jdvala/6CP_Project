
#for 2 classes Normal vs Attack
dataset1='C:/Users/Jay/Desktop/Project/classes_of_dataset/classes_for_KDD_test_data/class4.csv'


f1 = open('C:/Users/Jay/Desktop/Project/MachineLearning/dataset/NSL-KDD_Processed/NSL_TestLabels_int.csv', 'rt')
f2 = open(dataset1, 'wt')

for line in f1:
    if '0' in line:
        f2.write('0'+'\n')
    elif '1'in line:
        f2.write('1'+'\n')
    elif '2' in line:
        f2.write('2'+'\n')
    else:
        f2.write('3'+'\n')

f1.close()
f2.close()
