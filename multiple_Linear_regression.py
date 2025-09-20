import pandas as pd

df = pd.read_excel("C:/Users/Mahin/OneDrive/Documents/MachineLearning/Cars.xlsx")

print(df.head())

X = df[['weight', 'volume']]
y = df[['CO2']]

from sklearn.model_selection  import train_test_split
X_train , X_test , y_train , y_test  = train_test_split(X,y , test_size=0.3 , random_state= 42)

from sklearn.linear_model import LinearRegression

gf = LinearRegression()

gf.fit(X,y)

co2_pred = gf.predict(X_test)
print('This is the Co2_pred value for', X_test , 'is', co2_pred)

co2_pred2 = gf.predict([[1395,2500]])
print('This is the Co2_pred value:' , co2_pred2)

#If wanted plot later------------------------------------------

