
import pandas as pd

df = pd.read_csv("C:/Users/Mahin/OneDrive/Documents/MachineLearning/titanic_sink.csv")

df.drop(['PassengerId', 'Name', 'Ticket','SibSp','Cabin', 'Parch', 'Embarked'], axis= "columns" , inplace= True)

target = df.survive
inputs = df.drop('survive',axis= "columns")

dummies = pd.get_dummies(inputs.Sex)
# print(dummies.head(3))

inputs = pd.concat([inputs,dummies],axis='columns')
# print(inputs)

inputs.drop('Sex',axis='columns',inplace=True)
# print(inputs)

print(inputs.columns[inputs.isna().any()])

inputs.Age = inputs.Age.fillna(inputs.Age.mean())
# print(inputs.head())


from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test  = train_test_split(inputs ,target, test_size=0.3 , random_state= 42)

from sklearn.naive_bayes import GaussianNB

gn = GaussianNB()
gn.fit(X_train.values,y_train.values)

pclass = int(input('Enter the Class.No'))
age = int(input('Enter the age'))
gender = input('Enter the gender')

if gender == 'female':
    a = True
    b = False
else:
    a = False
    b = True

s = gn.predict([[pclass,age,a,b]])
print('------------------------------------------------------------------------')
if s == 1:
    print('The person has survived')
else:
    print('The person did not survive')


