# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.


"""
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn import set_config

#1. Read CSV data
file_path = r"C:\Users\naufa\OneDrive - Universiti Kebangsaan Malaysia\Desktop\shrdc ai technologist\deep learning\module\garments_worker_productivity.csv"
data = pd.read_csv(file_path)
data.head()

data.wip = data.wip.fillna(data.wip.mean())
data.head()

data.isnull().sum()


sns.distplot(data.actual_productivity)
#%%

p = PowerTransformer(method = 'box-cox')
data['actual_productivity'] = p.fit_transform(data[['actual_productivity']])
sns.displot(data.actual_productivity)

data.head()
data.shape
x = data.iloc[:,0:14]
x
y = data['actual_productivity']
y
#%%

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3)
x_train.shape
y_train.shape
x_test.shape
y_test.shape
data.head()

#%%
nom_cols = [0,3,5,6,7,8.9,10,14,15]

ord_cols = [1,2,4,11,12,13]

num_cols = [0,1,5,6,7,8,9,10,11,12,13,14,15]
#%%



trans = make_column_transformer((OneHotEncoder(sparse = False),nom_cols),
                                (OrdinalEncoder(), ord_cols)
                                ,(PowerTransformer(),num_cols),remainder = 'passthrough')
set_config(display= 'diagram')