#!/usr/bin/env python
# coding: utf-8

# # Dragon Real Estate -Price Predictor

# In[1]:


import pandas as pd


# In[2]:


housing=pd.read_csv("data.csv")


# In[3]:


housing.head()


# In[4]:


housing.info()


# In[5]:


housing['CHAS'].value_counts()


# In[6]:


housing.describe()


# In[7]:

# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


# #for plotting histograms
# import matplotlib.pyplot as plt
# housing.hist(bins=50,figsize=(20,15))


# # Train-Test Splitting

# In[9]:


# #for learning purpose
import numpy as np
# def split_train_test(data,test_ratio):
#     np.random.seed(42)
#     shuffled=np.random.permutation(len(data))
#     print(shuffled)
#     test_set_size=int(len(data)*test_ratio)
#     test_indices=shuffled[:test_set_size]
#     train_indices=shuffled[test_set_size:]
#     return data.iloc[train_indices],data.iloc[test_indices]
    


# In[10]:


# train_set,test_set=split_train_test(housing,0.2)


# In[11]:


# print(f"Rows in train set:{len(train_set)}\nRows in test set:{len(test_set)}")


# In[12]:


from sklearn.model_selection import train_test_split
train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42)
print(f"Rows in train set:{len(train_set)}\nRows in test set:{len(test_set)}")


# In[13]:


from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing['CHAS']):
    strat_train_set=housing.loc[train_index]
    strat_test_set=housing.loc[test_index]


# In[14]:


strat_train_set["CHAS"].value_counts()


# In[15]:


housing=strat_train_set.copy()


# # LOOKING FOR CORELATIONS

# In[16]:


corr_matrix=housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[17]:


from pandas.plotting import scatter_matrix
# attributes=["MEDV","RM","ZN","LSTAT"]
# scatter_matrix(housing[attributes],figsize=(12,8))


# In[18]:


housing.plot(kind="scatter",x="RM",y="MEDV",alpha=0.8)


# # Trying Out Attributes Combinations

# In[19]:


housing['TAXRM']=housing['TAX']/housing['RM']


# In[20]:


housing.head()


# In[21]:


corr_matrix=housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[22]:


housing.plot(kind="scatter",x="TAXRM",y="MEDV",alpha=0.8)


# In[23]:


housing=strat_train_set.drop('MEDV',axis=1)
housing_labels=strat_train_set['MEDV'].copy()


# # Missing Attributes

# In[24]:


# To take care of missing attributes,you have three options:
#     1.Get rid of the missing points
#     2.Get rid of the whole attributes
#     3.set the value to some value(0,mean or median)


# In[25]:


a=housing.dropna(subset=["RM"])#option 1
a.shape
#Note that there is no RM column and 
#also note that the original housing dataframe will remain unchanged


# In[26]:


housing.drop("RM",axis=1).shape #option2
#Note that there is no RM column and 
#also note that the original housing dataframe will remain unchanged


# In[27]:


median=housing["RM"].median()#compute median for option 3


# In[28]:


housing["RM"].fillna(median)#option3
#Note that there is no RM column and 
#also note that the original housing dataframe will remain unchanged


# In[29]:


housing.shape


# In[30]:


housing.describe()#before we started filling missing attributes


# In[31]:


from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy="median")
imputer.fit(housing)


# In[32]:


imputer.statistics_.shape


# In[33]:


X=imputer.transform(housing)


# In[34]:


housing_tr=pd.DataFrame(X,columns=housing.columns)


# In[35]:


housing_tr.describe()


# # Scikit-Learn Design

# Primarily,three types of objects:
# 
# 1.Estimators-It estimates some parameter based on parameter based on dataset.Eg imputer
# It has a fit method and transform method
# fit method-fits the dataset and calculates internal parameters
# 
# 2.Transformers-transform method takes input and returns output based on the learnings from fit().It also has a convenience function called fit_transform()
# 
# 3.Predictors-LinearRegression model is an example of predictor, fit() and predict() are two common functions. it also give score()
# function which will evaluate the predictions

# # Feature Scaling

# Primarily,two types of feature scaling methods:
# 1. Min-max scaling (Normalization)
#     (value-min)/(max-min)
#     Sklearn provides a class called MinMaxScaler for this
# 2. Standardization
#     (value-mean)/std
#     Sklearn provides a class called StandardScaler for this 
#     

# # Creating a Pipeline

# In[36]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    #   ....add as many as you want in your pipeline
    ('std_scaler',StandardScaler()),
])


# In[37]:


housing_num_tr=my_pipeline.fit_transform(housing)


# In[38]:


housing_num_tr.shape


# ## Selecting a desired model for Dragon Estates

# In[39]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()
# model=LinearRegression()
# model=DecisionTreeRegressor()
model.fit(housing_num_tr,housing_labels)


# In[40]:


some_data=housing.iloc[:5]


# In[41]:


some_labels=housing_labels.iloc[:5]


# In[42]:


prepared_data=my_pipeline.transform(some_data)


# In[43]:


model.predict(prepared_data)


# In[44]:


list(some_labels)


# ## Evaluating the model

# In[45]:


from sklearn.metrics import mean_squared_error
housing_predictions=model.predict(housing_num_tr)
mse=mean_squared_error(housing_labels,housing_predictions)
rmse=np.sqrt(mse)


# In[46]:


rmse


# ## Using better evaluation technique-cross validation

# In[47]:


from sklearn.model_selection import cross_val_score
scores=cross_val_score(model,housing_num_tr,housing_labels,scoring="neg_mean_squared_error",cv=10)
rmse_scores=np.sqrt(-scores)


# In[48]:


rmse_scores


# In[49]:


def print_scores(scores):
    print("Scores:",scores)
    print("Mean:",scores.mean())
    print("Standard deviation:",scores.std())


# In[50]:


print_scores(rmse_scores)


# ## Saving the model

# In[51]:


from joblib import dump, load
dump(model, 'Dragon.joblib')


# ## Testing the model on test data

# In[56]:


X_test=strat_test_set.drop("MEDV",axis=1)
Y_test=strat_test_set["MEDV"].copy()
X_test_prepared=my_pipeline.transform(X_test)
final_predictions=model.predict(X_test_prepared)
final_mse=mean_squared_error(Y_test,final_predictions)
final_rmse=np.sqrt(final_mse)
print(final_predictions,list(Y_test))


# In[53]:


final_rmse


# In[57]:


prepared_data[0]


# In[ ]:




