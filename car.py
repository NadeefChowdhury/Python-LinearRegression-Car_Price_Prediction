#import numpy as np, pandas as pd and matplotlib as plt. Remember to use plt.show() after every plot in case you're not using jupyter notebook
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

#read the csv data 
cars = pd.read_csv('H:\Machine learning\car data.csv')

#convert the fuel types to boolean values
cars2 = pd.get_dummies(cars['Fuel_Type'])
cars = pd.concat([cars, cars2], axis=1).reindex(cars.index)
cars.drop('Fuel_Type', axis=1, inplace=True)

#convert the seller types to boolean values
cars3 = pd.get_dummies(cars['Seller_Type'])
cars = pd.concat([cars, cars3], axis=1).reindex(cars.index)
cars.drop('Seller_Type', axis=1, inplace=True)

#divide the data into x and y. We're going to predict the selling price and so I have set y to selling price
y = cars['Selling_Price']
X = cars[['Year','Present_Price', 'Kms_Driven', 'CNG', 'Diesel', 'Petrol', 'Dealer',
       'Individual']]

#import train_test_split for splitting the data into training and testing groups
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)


#import LinearRegression, train the training data and use it to predict the selling prices of the testing data
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)
predictions = lm.predict(X_test)


#optional: create a scatter plot of y_test and predictions to see how accurate the model is
plt.scatter(y_test, predictions)


#check the accuracy
from sklearn import metrics
accuracy = metrics.explained_variance_score(y_test, predictions)
print('Accuracy in percentage: ', accuracy*100, '%')
