import pandas as pd
# import sklearn.feature_selection

df = pd.read_excel("C:/Users/Mahin/OneDrive/Documents/MachineLearning/ham_spam_messages.xlsx")

df['Is_Spam'] = df['Category'].apply(lambda x: 0 if x == 'ham' else 1)

X = df['Message']
y = df['Is_Spam']

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train,y_train)

while True:
    d = input('Enter the message')
    model_test =[d]
    model_v = vectorizer.transform(model_test)
    res = model.predict(model_v)
    if res == 1:
        print('There is spam in this message')
    else:
        print("There is no spam in this message")

