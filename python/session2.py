# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 12:16:58 2019

@author: BALJEET
"""
#data wrangling

from pandas import read_csv
import pandas as pd

path = r"C:\Users\anujk\Documents\tennis.csv"
dataframe = read_csv(path)
yes=dataframe[dataframe['play']=='Yes']
no=dataframe[dataframe['play']=='No']

psunny_yes=len(yes[yes['outlook']=="Sunny"])/len(yes)
phot_yes = len(yes[yes['temp']=="Hot"])/len(yes)
phumidity_yes = len(yes[yes['humidity']=="High"])/len(yes)
pwind_yes = len(yes[yes['wind']=="Strong"])/len(yes)

psunny_no=len(no[no['outlook']=="Sunny"])/len(no)
phot_no = len(no[no['temp']=="Hot"])/len(no)
phumidity_no = len(no[no['humidity']=="High"])/len(no)
pwind_no = len(no[no['wind']=="Strong"])/len(no)

(psunny_yes + phot_yes + phumidity_yes + pwind_yes )

dataframe.head(5)
dataframe.shape
dataframe.describe()

#individual data or slices
dataframe.iloc[0]
dataframe.iloc[10]

dataframe.iloc[-1]
# 2 3 4 rows
dataframe.iloc[1:4]
# upto 4th row
dataframe.iloc[:4]

#conditional statement
dataframe[dataframe['Sex']=='female']




(dataframe['Sex']=='female') & (dataframe['Age']>=65)
dataframe[(dataframe['Sex']!='female') & (dataframe['Age']>=65)]


#replace
dataframe['Sex'].replace("female","male")



dataframe.replace(r"Mr","Mr.",regex=True)

#min max sum ,avg etc

print("Minimum",dataframe['Age'].min())
dataframe['Age'].max()
dataframe['Age'].mean()
dataframe['Age'].sum()
dataframe['Age'].count()
dataframe.count()


#unique
dataframe['Sex'].unique()
dataframe['Sex'].value_counts()
dataframe['PClass'].value_counts()
dataframe['Sex'].nunique()
dataframe['PClass'].nunique()


dataframe[dataframe['PClass'] == '*']
#missing values
dataframe[dataframe['Age'].isnull()].head(5)



import numpy as np
dataframe['Sex'].replace('male',np.nan)

#deleting a column
dataframe.drop('Age')
dataframe.drop('Age',axis=1)
dataframe.drop(['Age','Sex'],axis=1)
dataframe.drop(dataframe.columns[1],axis=1)

#deleting a row using a boolean condition
dataframe[dataframe['Sex']!='male'].head(2)
dataframe.drop([0, 1],axis=0)  #working
dataframe[dataframe.index!=0]


#Group
dataframe.groupby('Sex').mean()
dataframe.groupby('Sex').count()
dataframe.groupby('Survived').count()
#looping over a column
for name in dataframe['Name'][0:2]:
    print( name.upper())

for age in dataframe['Age']:
    print( age+1)
    
    
#applying a function
def uppercase(x):
    print( x.upper())

for name in dataframe['Name'][0:2]:
    uppercase(name)


dataframe['Name'].apply(uppercase)[0:2]


#apply lambda function to group

dataframe.groupby('Sex').apply(lambda x: x.count())    

#concatenate dataframes by rows

pd.concat([dataframe1,dataframe2],axis=0)

#concatenate dataframes by cols
pd.concat([dataframe1,dataframe2],axis=1)

pd.
#creating a dataframe

import pandas as pd

dataframe=pd.DataFrame()

dataframe['Name']=['Baljeet Kaur', 'Aman Ahuja']
dataframe['Roll Num']=[111,222]
dataframe['Hostel']=[True, False]