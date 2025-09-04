#!/usr/bin/env python
# coding: utf-8

# In[42]:


#Data science and technology,machine learning project (MAJORING ON LINEAR REGRESSION)
#Importing the Dependencies 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt#for visualization
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn import metrics


# In[43]:


#Data Collection and Analysis


# In[92]:


#loading the data from csv file to a pandas dataflame
insurance_dataset=pd.read_csv("insurance[1].csv")


# In[93]:


#printing first 5 rows of the dataset
insurance_dataset.head()


# In[94]:


#printing last 5 rows of the dataset
insurance_dataset.tail()


# In[95]:


#checking no of rows and columns of the dataset
insurance_dataset.shape


# In[96]:


#Getting some information about the dataset
insurance_dataset.info()


# In[97]:


#our categorical features:
#..Sex
#..Smoker
#..Region


# In[98]:


#checking for missing values
insurance_dataset.isnull()


# In[99]:


#checking for missing values
insurance_dataset.isnull().sum()


# In[100]:


#Data Analysis


# In[101]:


#Statistical measures about the dataset
insurance_dataset.describe()


# In[102]:


#Statistical measures about the dataset;give sum of each column
insurance_dataset.describe().sum()


# In[103]:


#Finding the Distribution of Dataset


# In[104]:


#Distribution of ...age.. value(age is a has large numerical value thus use distplot)
sns.set()
plt.figure(figsize=(6,6))
sns.displot(insurance_dataset['age'], kde=True)  # `kde=True` adds the smooth line
plt.title('Age Distribution')
plt.show()


# In[105]:


#Distribution of ...Gender column.. value
plt.figure(figsize=(6,6))
sns.countplot(x='sex',data=insurance_dataset)
plt.title('Sex Distribution')
plt.show()


# In[106]:


#checking datapoints for each sex column values
insurance_dataset['sex'].value_counts()


# In[107]:


#bmi distribution of datasets
plt.figure(figsize=(6,6))
sns.displot(insurance_dataset['bmi'], kde=True)  # `kde=True` adds the smooth line
plt.title('bmi Distribution')
plt.show()


# In[108]:


#conclusion from bmi distribution
#Normal BMI Range..>18.5 to 24.9 implies that if the bmi of a person is less than 18.5 then that person is underweight $
#..if the bmi is over 24.9 then that person is overweight.Alot of people in the dist are overweight.


# In[109]:


#Distribution of ...Children column.. value
plt.figure(figsize=(6,6))
sns.countplot(x='children',data=insurance_dataset)
plt.title('Children Distribution')
plt.show()


# In[110]:


#checking datapoints for each children column values
insurance_dataset['children'].value_counts()


# In[111]:


#Distribution of ...smoker column.. value
plt.figure(figsize=(3,3))
sns.countplot(x='smoker',data=insurance_dataset)
plt.title('smoker Distribution')
plt.show()


# In[112]:


#checking datapoints for each smoker column values
insurance_dataset['smoker'].value_counts()


# In[113]:


#Distribution of ...region column.. value
plt.figure(figsize=(6,6))
sns.countplot(x='region',data=insurance_dataset)
plt.title('region Distribution')
plt.show()


# In[114]:


#checking datapoints for each region column values
insurance_dataset['region'].value_counts()


# In[115]:


#Distribution of ...charges.. value(charges is a has large numerical value thus use distplot)
sns.set()
plt.figure(figsize=(6,6))
sns.displot(insurance_dataset['charges'], kde=True)  # `kde=True` adds the smooth line
plt.title('charges Distribution')
plt.show()


# In[116]:


#Data Pre_processing

#Encoding the categorical features


# In[124]:


#encoding 'sex' column
insurance_dataset.replace({'sex':{'male':0,'female':1}},inplace=True)
#encoding 'smoker' column
insurance_dataset.replace({'smoker':{'yes':0,'no':1}},inplace=True)
#encoding 'region' column
insurance_dataset.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}},regex=False)
# inplace=True is used to modify the DataFrame directly for 'sex' and 'smoker' columns.
# regex=False ensures exact string matching for 'region' values, avoiding unintended pattern matching..e.g southeast $ southwest
#..we have two souths and northeast # northwest we have two norths


# In[129]:


#splitting the features and target


# In[130]:


x=insurance_dataset.drop(columns='charges',axis=1)#target variable
#axis=1 for columns drop $axis=0 for rows drop
y=insurance_dataset['charges']#y stores target variable


# In[131]:


print(x)


# In[132]:


print(y)


# In[133]:


#splitting the data into Training data $Testing data


# In[135]:


x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2,random_state=2)
#test_size=0.2 means how much data you ant in testing data e.g in our case 0.2 means 20%
#random_state can be chosen randomly i.e 2,3,4...etc


# In[136]:


print(x.shape,x_train.shape,x_test.shape)


# In[141]:


#MODEL TRAINING

#linear regression


# In[142]:


#loaing the linear regressionb model
regressor=LinearRegression()


# In[143]:


regressor.fit(x_train,y_train)


# In[144]:


#Model Evaluation


# In[145]:


#prediction on training data
training_data_prediction=regressor.predict(x_train)


# In[147]:


# R_Squared value
r2_train=metrics.r2_score(y_train,training_data_prediction)
print('R squared value:',r2_train)


# In[148]:


#prediction on test data
test_data_prediction=regressor.predict(x_test)


# In[149]:


# R_Squared value
r2_test=metrics.r2_score(y_test,test_data_prediction)
print('R squared value:',r2_test)


# In[150]:


#CONCLUSION For R Squared Value for both training and testing data
#..Values almost equal to each other i.e 0.75 $ 0.74 thus no overfitting issue
#NB..R Squared for training should be larger than for testing


# In[151]:


#Buliding a predictive system


# In[174]:


# input_data=(31,female,25.74,0,no,southeast)..i.e replace female,no $ southeast with 1,0 $ 0 as done above in Data pre_preprocessing..
#...while encoding the categorical features
#input_data (cateory features on the left,any row,only $ leave target variable 3756.6216 i.e charges) above from csv opened by notepad
input_data=(31,1,25.74,0,1,0)
# Convert to DataFrame with proper feature names
input_data = pd.DataFrame([input_data], columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])
# Predict
prediction = regressor.predict(input_data)
print(prediction)
print('The insurance cost is USD',prediction[0])


# In[ ]:


#CONCLUSION..The insurance cost(predicted cost) 3760 is close to real cost 3756.
#The predicted insurance cost of a person is USD 3760

