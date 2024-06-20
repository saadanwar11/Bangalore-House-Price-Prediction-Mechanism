#!/usr/bin/env python
# coding: utf-8
# In[1]:

import pandas as pd
import numpy as np

# In[3]:
data = pd.read_csv('Bengaluru_House_Data.csv')

data.head()

data.shape

data.describe()

data.info

for column in data.columns:
    print(data[column].value_counts())
    print("*"*20)

data.isna().sum()

data.drop(columns=['area_type','availability','society','balcony'],inplace = True)

data.describe()

data.info

data['location'].value_counts()

data['location'] = data['location'].fillna('Sarjapur Road')

data['size'].value_counts()

data['size'] = data['size'].fillna('2 BHK')

data['bath'] = data['bath'].fillna(data['bath'].median())

data.info()

data['bhk'] = data['size'].str.split().str.get(0).astype(int)

data[data.bhk > 20]

data['total_sqft'].unique()

def is_float(x):
    try:
        float(x)
    except:
        return False
    return True
def convertRange(x):
    temp = x.split('-')
    if len(temp) == 2:
        return (float(temp[0]) + float(temp[1]))/2
    try:
        return float(x)
    except:
            return None

data['total_sqft'] = data['total_sqft'].apply(convertRange)

data.head()

data['price_per_sqft'] = data['price']*100000/data['total_sqft']

data['price_per_sqft']

data.describe()

data['location'].value_counts()

location_counts_less_10 = location_counts[location_counts <= 10]
location_counts_less_10

data['location'] = data['location'].apply(lambda x:x.strip())
location_counts = data['location'].value_counts()

location_counts


data['location']=data['location'].apply(lambda x: 'other' if x in location_counts_less_10 else x)

data.describe()

(data['total_sqft']/data['bhk']).describe()

data = data[((data['total_sqft']/data['bhk']) >= 300)]

data.describe()

data.shape

data.price_per_sqft.describe()

def remove_outliers_sqft(df):
    df_output = pd.DataFrame()
    for key,subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)

        gen_df = subdf[(subdf.price_per_sqft > (m-st)) & (subdf.price_per_sqft <= (m+st))]
        df_output = pd.concat([df_output,gen_df], ignore_index = True)

    return df_output
data = remove_outliers_sqft(data)
data.describe()

def bhk_outlier_remover(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean':np.mean(bhk_df.price_per_sqft),
                'std' : np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
            return df.drop(exclude_indices, axis = 'index')

data=bhk_outlier_remover(data)

data.shape

data

data.drop(columns=['size','price_per_sqft'],inplace=True)

data.head()

data.to_csv("cleaned_data.csv")

x = data.drop(columns=['price'])
y = data['price']

# from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state=0)

print(x_train.shape)
print(y_train.shape)

column_trans = make_column_transformer((OneHotEncoder(sparse = False),['location']),remainder = 'passthrough')
scaler = StandardScaler()
lr = LinearRegression(normalize=True)
pipe = make_pipeline(column_trans,scaler)
pipe.fit(x_train,y_train)

# # apply lasso

lasso = Lasso()

pipe = make_pipeline(column_trans,scaler,lasso)

pipe.fit(x_train,y_train)

y_pred_lasso = pipe.predict(x_test)
r2_score(y_test,y_pred_lasso)

ridge = Ridge

pipe = make_pipeline(column_trans,scaler,ridge)

import pickle

pickle.dump(pipe, open('RidgeModel.pkl', 'wb'))
