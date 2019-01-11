# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 10:19:08 2018

@author: 555224
"""
#https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy
###PYTHON DATA WRANGLING

#####

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

TRAIN=pd.read_csv("D:\\R datasets\\titanic\\train.csv")
TEST=pd.read_csv("D:\\R datasets\\titanic\\test.csv")


TRAIN.head()



TRAIN.describe(include='all')

###### AGE, EMBARKED, CABIN HAS MISSING DATA

TRAIN.info()

COMBINE=[TRAIN,TEST]

TRAIN.columns

######### cross table

pd.crosstab(TRAIN['Sex'],TRAIN['Pclass'])

############ DIRECT ASSIGNMENT DOESN'T CREATE A COPY
TRAIN_C=TRAIN.copy(deep=True)

############ GROUP BY COMMAND 
TRAIN[['Pclass','Sex','Age']].groupby(by=['Pclass','Sex'],axis=0,as_index=False).mean()


########### CONDITIONAL FILTER

 TRAIN.loc[TRAIN['Age'].isnull(),'Age']


for dataset in COMBINE:
    dataset.loc[dataset['Age'].isnull(),'Age']=10
    
########### REPLACE COMMAND
    
TRAIN.replace('LOL',np.nan,inplace=True)
    

########## MISSING DATA
    missing_data = TRAIN.isnull()

#############
    
TRAIN['Pclass'].value_counts()    
 
############# ARRAY CONVERSION   
ARR=TRAIN.values


# simply drop whole row with NaN in "price" column
TRAIN.dropna(subset=["Embarked"], axis=0, inplace = True)

# reset index, because we droped two rows
TRAIN.reset_index(drop = True, inplace = True)

###### c=Change the format of the columns
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")


######### GENERATING THE DUMMNY VARIABLES FOR CATEGORICAL VARS
INT1=pd.get_dummies(TRAIN["Sex"])


######### DATA FRAME CONCATINATION
INT2=pd.concat([TRAIN,INT1],axis=0)

########
df['drive-wheels'].unique()

########## CORRELATION BETWEEN VARIABLES OF DATA FRAME
df.corr()




######### EXPORT CSV
INT2.to_csv('clean_df.csv')






########### SCIPY PACKAGE

from scipy import stats

filename = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
df = pd.read_csv(filename, names = headers)
df.head()


########correlation
pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  


df.replace('?',np.nan,inplace=True)

######### ANOVA

df_gptest.dropna(axis=0,inplace=True)

df_gptest=df[['drive-wheels','body-style','price']].copy(deep=True)
grouped_test2=df_gptest[['drive-wheels','price']].groupby(['drive-wheels'])
grouped_test2.head(2)

grouped_test2.get_group('4wd')['price']


f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'], grouped_test2.get_group('4wd')['price'])  
 
print( "ANOVA results: F=", f_val, ", P =", p_val) 


######## BINNING THE DATASET qcut is based on percentiles and cut is based on mean

df['BinTest']=pd.qcut(df['normalized-losses'].astype('float'),4)

df['Bi1nTest']=pd.cut(df['normalized-losses'].astype('float'),4)

df[['Bi1nTest','BinTest']]


###COMPLETING: complete or delete missing values in train and test/validation dataset
for dataset in data_cleaner:    
    #complete missing age with median
    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)

    #complete embarked with mode
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)

    #complete missing fare with median
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)
    
#delete the cabin feature/column and others previously stated to exclude in train dataset
drop_column = ['PassengerId','Cabin', 'Ticket']
data1.drop(drop_column, axis=1, inplace = True)

print(data1.isnull().sum())
print("-"*10)
print(data_val.isnull().sum())



###############LOC Function

###CREATE: Feature Engineering for train and test/validation dataset
for dataset in data_cleaner:    
    #Discrete variables
    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1

    dataset['IsAlone'] = 1 #initialize to yes/1 is alone
    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1

    #quick and dirty code split title from name: http://www.pythonforbeginners.com/dictionary/python-split
    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    
    
########### PREPROCESSING FUNCTIONS
from sklearn.preprocessing import LabelEncoder    

label=LabelEncoder()

TRAIN['SEXCODE']=label.fit_transform(TRAIN['Sex'])
TRAIN['Embarked_Code'] = label.fit_transform(TRAIN['Embarked'])


from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1])


from sklearn.preprocessing import StandardScaler
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

########### TEST _TRAIN SPLIT

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

############ CROSS VALIDATION THE MODEL
from sklearn.model_selection import cross_val_score

value=cross_val_score(alg, dtrain[predictors], dtrain[target], cv=20, scoring='mean_squared_error')
cv= K value of the cross validation
scoring= type of result i.e. mean squared error, or f1_score or r2 score
alg=Linear_Regression(),svm.SVC(kernel='Linear',C=1)

########### RANDOM_FOREST_CLASSIFIER
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)
