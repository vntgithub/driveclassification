import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def Acc(matrix, i):
	x = 0;
	for j in range(0, 11):
		x = x + matrix[i][j];
	return matrix[i][i]/x;

def count(dt):
	Y = 1
	count = 0
	for i in range(0, 58509):
		if dt[i] == Y:
			count = count + 1
		else:
			print("Nhan ", Y, " co so luong = ", count)
			Y = Y + 1
			count = 0	
	print("Nhan ", Y, " co so luong = ", count)


#Doc du lieu
dt = pd.read_csv("Sensorless_drive_diagnosis.txt", delimiter = " ", header=None);


X = dt.iloc[:, 0:48]
Y = dt.iloc[:, 48]
count(Y)

#------------------------------------------------------------------------------
#Cau b
print("So phan tu cua tap du lieu: ", len(X));
print("So nhan cua du lieu: ", np.unique(Y)) # So nhan cua lap du lieu la 11: 1-11;

#------------------------------------------------------------------------------

#Cau c

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 100);

print("So nhan cua tap test: ", np.unique(y_test));
#np.unique(y_test)#So nhan cua tap test la 7: 3, 4, 5, 6, 7, 8, 9

#------------------------------------------------------------------------------

#Cau d

            #-----------------GINI--------------------
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 10, max_depth = 15, min_samples_leaf = 5);
clf_gini.fit(X_train, y_train);
y_pred = clf_gini.predict(X_test);
            #-----------------ENTROPY------------------

#------------------------------------------------------------------------------

#Do chinh xac
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
    #Do chinh xac tong the
print("Accuracy = ", accuracy_score(y_test, y_pred)*100); #Accuracy = 98,26
    #Do chinh xac tung lop
matrix = confusion_matrix(y_test, y_pred);
print(matrix)
for i in range(0,11):
	print("Do chinh xac phan lop", i+1, "la: ", Acc(matrix, i));
#------------------------------------------------------------------------------

#Cau f: Do chinh xac 6 phan tu cua tap test
    #Do chinh xac tong the
print("Accuracy is ", accuracy_score(y_test.iloc[0:6], y_pred[0:6])*100); #Accuracy = 50
    #Do chinh xac tung lop
matrix2 = confusion_matrix(y_test.iloc[0:6], y_pred[0:6]);
print(matrix2)


#------------------------------------------------------------------------------
#Bayes 
model = GaussianNB()
model.fit(X_train, y_train)
y_pred_Bayes = model.predict(X_test)
print("Accuracy (Bayes) = ", accuracy_score(y_test, y_pred_Bayes)*100)
matrix_Bayes = confusion_matrix(y_test, y_pred_Bayes);
print(matrix_Bayes)
for i in range(0,11):
	print("Do chinh xac phan lop", i+1, "la: ", Acc(matrix_Bayes, i));