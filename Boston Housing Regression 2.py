#import Library
import pandas as pd

import numpy as np

from sklearn.feature_selection import GenericUnivariateSelect

from sklearn.feature_selection import chi2

#*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*

#Read data & show details

path = "C:/Users/Khaled/Desktop/coding\python/Machine Learning new/Applications/house-prices-advanced-regression-techniques/train.csv"

data_train = pd.read_csv(path)

pd.set_option("display.max_columns" , None)

pd.set_option("display.max_row" , None)

print("data is \n" , data_train.head(10))

print("data shape is \n" , data_train.shape)

print(" columns data is \n" , data_train.columns)

print(" describe data is \n" , data_train.describe())

#*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*

#Separation of data

X_train = data_train.iloc[: , 0 : -1] 

print("data shape is \n" , X_train.shape)

X_train.head(10)

y_train = data_train.iloc[: , -1]

print("data shape is \n" , y_train.shape)

y_train.head(10)

y_train.describe()


#*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*

#Remove NAN from data

#from sklearn.impute import SimpleImputer

#SimpleImputerModel = SimpleImputer(missing_values=np.nan , strategy="mean")
#
#del_NAN=SimpleImputerModel.fit_transform(X_train)
#
#print(del_NAN)

total = X_train.isnull().sum().sort_values(ascending=False)

percent = (X_train.isnull().sum() / X_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total ,percent ] , axis=1 , keys=["total" , "percent"])

print(" Missing data =\n " , missing_data.head(20))

New_X_train = X_train.drop((missing_data[missing_data["total"] > 1]).index,1)

New_X_train = New_X_train.drop(New_X_train.loc[New_X_train["Electrical"].isnull()].index)

New_X_train.isnull().sum().max()

print("New_X_train =\n" , New_X_train.head(10))


#*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*

#from sklearn.preprocessing import OneHotEncoder
#
#OneHotEncoderModel = OneHotEncoder()
#
#OneHotEncoderModel.fit_transform(New_X_train).toarray()
#
#print("New_X_train =\n" , New_X_train.head(10))

New_X_train = pd.get_dummies(New_X_train)

print("New_X_train =\n" , New_X_train.head(2))

print("New_X_train =\n" , New_X_train.shape)

print(" columns data is \n" , New_X_train.columns)


#*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*-*_*

#feature_selection with Generic Univariate Select 

#GenericUnivariateSelectModel = GenericUnivariateSelect( chi2 , "k_best"  , param=5)
#
#GenericUnivariateSelectModel.fit(New_X_train , y_train )








































































































































































































