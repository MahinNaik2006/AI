import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_excel("C:/Users/Mahin/OneDrive/Documents/MachineLearning/height_weight_class_dataset.xlsx")
print(df)

# sd = df.drop('Class',axis='columns')
# print(sd)

df['Class_No'] = df['Class'].map({
    'Underweight': 0,
    'Normal': 1,
    'Overweight': 2
})


X = df[['Height_cm','Weight_kg']]
y = df[['Class_No']]

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.4)

import math
s = int(math.sqrt(len(y_test)))
s = s + 1

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors = s , metric= 'euclidean')

model.fit(X_train.values,y_train.values.ravel())

h = float(input('Enter the height'))
w = float(input('Enter the Weight'))

prediction = model.predict([[w,h]])

if prediction == 0:
    print('UnderWeight')
elif prediction == 1:
    print('Normal')
else:
    print('OverWeight')