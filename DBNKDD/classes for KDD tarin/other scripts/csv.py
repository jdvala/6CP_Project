import csv

f1 = open ('C:/Users/Jay/Desktop/MachineLearning/dataset/NSL-KDD_Processed/NSL_ValidLabels_int.csv','r')

try: 
	writer = csv.writer('C:/Users/Jay/Desktop/MachineLearning/dataset/NSL-KDD_Processed/NSL_ValidLabels_int.csv','w')
 	for line in f1:
 		if '1' in line:
 			writer.writerow('1','0','0','0','0',+'\n')
 		elif '2' in line:
 			writer.writerow('0','1','0','0','0',+'\n')
 		elif '3' in line:
 			writer.writerow('0','0','1','0','0',+'\n')
 		elif '4' in line:
 			writer.writerow('0','0','0','1','0',+'\n')
 		elif '5' in line:
 			writer.writerow('0','0','0','0','1',+'\n')	
finally:
	f1.close()