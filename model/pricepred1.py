import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

dataset1= pd.read_csv("Bengaluru_House_Data.csv")

print(dataset1['area_type'])
x1=dataset1.groupby('area_type')['area_type'].agg('count')



#Dropping Unneccesary Coloumns
x2=dataset1.drop(['area_type','society','balcony','availability'],axis='columns')
x2.head()

#Cleaning the Dataset, looking for NA-Values
x2.isnull().agg('sum')

#Handling NA VALUES USING MEDIAN
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values = np.nan, strategy = 'median', verbose = 0)
imputer=imputer.fit(x2[['bath']])
x2['bath']=imputer.transform(x2[['bath']]).ravel()
                            
#Dropping the few left NA Values 
x3=x2.dropna()
x3.isnull().sum()



#Creating New Coloumn spliting the data in size
x3['size'].unique()
x3['bedroom']=x3['size'].apply(lambda x: int(x.split(' ')[0]))

x3['size'].unique()

#Handling IntervalValues
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

x3[~x3['total_sqft'].apply(is_float)]

def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens)==2:
        return(float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None

x4=x3.copy()
x4['total_sqft']=x4['total_sqft'].apply(convert_sqft_to_num)
x4.loc[30]
    
    
#Handling Categorical data and preparing dataset for Outlier detection
x5= x4.copy()
x5['price_per_sqft']=x5['price']*100000/x5['total_sqft']

len(x5.location.unique())
loc_stats=x5.groupby('location')['location'].agg('count').sort_values(ascending=False)


len(loc_stats[loc_stats<=10])

##It can be seen most of the location has an occurence of 1 and less than 10, it would be better to consider them into 1

loc_stats_less_than_10=loc_stats[loc_stats<=10]


x5.location = x5.location.apply(lambda x: 'other' if x in loc_stats_less_than_10 else x)

#Removing Outliers
#1 bhk will require an area of 300sqft min, based on which we are removing the outliers
x5[x5.total_sqft/x5.bedroom<300].head()


x6=x5[~(x5.total_sqft/x5.bedroom<300)]
x6.shape              #Removed some outlier




def remove_pps_outliers(df):
    df_out=pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m=np.mean(subdf.price_per_sqft)
        st=np.std(subdf.price_per_sqft)
        reduced_df=subdf[(subdf.price_per_sqft>(m-st))&(subdf.price_per_sqft<=(m+st))]
        df_out=pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
x7=remove_pps_outliers(x6)

#plotting histogram to see teh curve
import matplotlib
matplotlib.rcParams["figure.figsize"]=(20,10)
plt.hist(x7.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Sqft")
plt.ylabel("Count")


x8=x7[x7.bath<x7.bedroom+2]



#plotting histogram to see teh curve
import matplotlib
matplotlib.rcParams["figure.figsize"]=(20,10)
plt.hist(x8.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Sqft")
plt.ylabel("Count")



#Dropping the coloumns which we created for data manipulation and cleansing purposes

x9=x8.drop(['size','price_per_sqft'],axis='columns')


#Handling categorical Data

#from sklearn.preprocessing import OneHotEncoder 
#rom sklearn.compose import ColumnTransformer 
   



# creating one hot encoder object with categorical feature 0 
# indicating the first column 

#transformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')

  
#x9 = np.array(transformer.fit_transform(x9), dtype = np.str)

dummies=pd.get_dummies(x9.location)

#Concating two column and dropping other from locations
x10=pd.concat([x9,dummies.drop('other', axis='columns')],axis='columns')

#removing the location coloumn now as we have encoded it
x11=x10.drop('location',axis='columns')

#Dividing the data into training and test 
X=x11.drop('price',axis='columns')

Y=x11.price

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size=0.2, random_state=10)

#Building Model
from sklearn.linear_model import LinearRegression
lr_clf=LinearRegression()
lr_clf.fit(X_train, Y_train)
lr_clf.score(X_test, Y_test)





#Price prediction using the Model
def  predict_price(location,sqft,bath,bedroom):
    loc_index=np.where(X.columns==location)[0][0] #locating appropriate column
    
    x=np.zeros(len(X.columns))
    x[0]=sqft
    x[1]=bath
    x[2]=bedroom
    if loc_index>=0:
        x[loc_index]=1
        return lr_clf.predict([x])[0]
    
predict_price('Vishveshwarya Layout', 1000, 2, 2)

#Importing Model and columns
import pickle
with open('bangalore_real_estate_price_prediction.pickle','wb') as f:
    pickle.dump(lr_clf, f)

import json
columns={
    'data_columns': [col.lower() for col in X.columns]}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))




















