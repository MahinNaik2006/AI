import  pandas as pd
import warnings

s = pd.read_excel('C:/Users/Mahin/OneDrive/Documents/MachineLearning/payment_credit_score_dataset_large.xlsx')

print(s.head())

A = s[['Initial Payment' , 'Final Payment' ,  'Credit Score']]
B = s[['Result']]

from sklearn.model_selection import train_test_split
A_train , A_test , B_train , B_test = train_test_split(A,B,test_size=0.3 , random_state=42 )

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

dtc.fit(A_train.values , B_train.values)

pred = dtc.predict(A_test.values)
print(pred)

pred2 = dtc.predict([[5167,8132,772]])
print('Should the money be given', pred2)

