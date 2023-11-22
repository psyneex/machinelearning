import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error


df = pd.read_csv("/Users/h4mi6/Documents/Datasets/Melbourne_housing_FULL.csv")
pw = df.head()

#Scrubbing

#removing unnessesary columns 
df.drop(['Address','Method','SellerG','Date','Postcode','Lattitude','Longtitude','Regionname','Propertycount'], axis=1, inplace=True)

#removing rows with missing values
df.dropna(axis=0,how='any',subset=None,inplace=True)

#one-hot encoding
df = pd.get_dummies(df,columns=['Suburb','CouncilArea','Type'])    

#splitting the dataset for train and test
x = df.drop('Price',axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3,shuffle=True)

#setting up the algorithm heyperparameters
sample_model = ensemble.GradientBoostingRegressor(

        n_estimators=20,           #states the number of decision trees.
        learning_rate=0.3,          #controls the rate at which additional decision trees influence the overall prediction
        max_depth=15,               #defines the maximum number of layers(depth) for each decision tree
        min_samples_split=4,        #defines the minimum number of samples required to execute a new binary split
        min_samples_leaf=6,         #represents the minimum number of samples that must appear in each child node (leaf) before a new branch can be implemented
        max_features=0.6,           #is the total number of features presented to the model when determining the best split
        loss='huber'                #model's error rate

 ) 

#linking the training data to the learning algorithm
sample_model.fit(X_train,y_train)        

#evaluating the performance
mean_train = mean_absolute_error(y_train,sample_model.predict(X_train))      
mean_test = mean_absolute_error(y_test,sample_model.predict(X_test))

print("training set mean absolute error:  "+ str(mean_train))
print("test set mean absolute error:  "+ str(mean_test))

print('done!')
