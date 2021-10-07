# Dragon Real Estate -Price Predictor
import pandas as pd
housing=pd.read_csv("data.csv")
housing.head()
housing.info()
housing['CHAS'].value_counts()
housing.describe()

# %matplotlib inline
# #for plotting histograms
# import matplotlib.pyplot as plt
# housing.hist(bins=50,figsize=(20,15))

# Train-Test Splitting
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
# train_set,test_set=split_train_test(housing,0.2)
# print(f"Rows in train set:{len(train_set)}\nRows in test set:{len(test_set)}")
from sklearn.model_selection import train_test_split
train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42)
print(f"Rows in train set:{len(train_set)}\nRows in test set:{len(test_set)}")
from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing['CHAS']):
    strat_train_set=housing.loc[train_index]
    strat_test_set=housing.loc[test_index]
strat_train_set["CHAS"].value_counts()
housing=strat_train_set.copy()

# LOOKING FOR CORELATIONS
corr_matrix=housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)
from pandas.plotting import scatter_matrix
# attributes=["MEDV","RM","ZN","LSTAT"]
# scatter_matrix(housing[attributes],figsize=(12,8))
housing.plot(kind="scatter",x="RM",y="MEDV",alpha=0.8)

# Trying Out Attributes Combinations
housing['TAXRM']=housing['TAX']/housing['RM']
housing.head()
corr_matrix=housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)
housing.plot(kind="scatter",x="TAXRM",y="MEDV",alpha=0.8)
housing=strat_train_set.drop('MEDV',axis=1)
housing_labels=strat_train_set['MEDV'].copy()

# Missing Attributes
# To take care of missing attributes,you have three options:
#     1.Get rid of the missing points
#     2.Get rid of the whole attributes
#     3.set the value to some value(0,mean or median)
a=housing.dropna(subset=["RM"])#option 1
a.shape
#Note that there is no RM column and 
#also note that the original housing dataframe will remain unchanged
housing.drop("RM",axis=1).shape#option2
#Note that there is no RM column and 
#also note that the original housing dataframe will remain unchanged
median=housing["RM"].median()#compute median for option 3
housing["RM"].fillna(median)#option3
#Note that there is no RM column and 
#also note that the original housing dataframe will remain unchanged
housing.shape
housing.describe()#before we started filling missing attributes
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy="median")
imputer.fit(housing)
imputer.statistics_.shape
X=imputer.transform(housing)
housing_tr=pd.DataFrame(X,columns=housing.columns)
housing_tr.describe()

# Scikit-Learn Design
# Primarily,three types of objects:

# 1.Estimators-It estimates some parameter based on parameter based on dataset.Eg imputer It has a fit method and transform method fit method-fits the dataset and calculates internal parameters

# 2.Transformers-transform method takes input and returns output based on the learnings from fit().It also has a convenience function called fit_transform()

# 3.Predictors-LinearRegression model is an example of predictor, fit() and predict() are two common functions. it also give score() function which will evaluate the predictions

# Feature Scaling
# Primarily,two types of feature scaling methods:

# 1.Min-max scaling (Normalization) (value-min)/(max-min) Sklearn provides a class called MinMaxScaler for this
# 2.Standardization (value-mean)/std Sklearn provides a class called StandardScaler for this

# Creating a Pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    #   ....add as many as you want in your pipeline
    ('std_scaler',StandardScaler()),
])
housing_num_tr=my_pipeline.fit_transform(housing)
housing_num_tr.shape
# Selecting a desired model for Dragon Estates
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# model=RandomForestRegressor()
# model=LinearRegression()
model=DecisionTreeRegressor()
model.fit(housing_num_tr,housing_labels)
some_data=housing.iloc[:5]
some_labels=housing_labels.iloc[:5]
prepared_data=my_pipeline.transform(some_data)
model.predict(prepared_data)
list(some_labels)

# Evaluating the model
from sklearn.metrics import mean_squared_error
housing_predictions=model.predict(housing_num_tr)
mse=mean_squared_error(housing_labels,housing_predictions)
rmse=np.sqrt(mse)
rmse

# Using better evaluation technique-cross validation
from sklearn.model_selection import cross_val_score
scores=cross_val_score(model,housing_num_tr,housing_labels,scoring="neg_mean_squared_error",cv=10)
rmse_scores=np.sqrt(-scores)
rmse_scores
def print_scores(scores):
    print("Scores:",scores)
    print("Mean:",scores.mean())
    print("Standard deviation:",scores.std())
    with open("comp.txt","a") as f:
        f.write(f"{model}\n\tMean:{scores.mean()}\n\tStandard deviation:{scores.std()}") 
print_scores(rmse_scores)

# Saving the model
from joblib import dump, load
dump(model, 'Dragon.joblib')
X_test=strat_test_set.drop("MEDV",axis=1)
Y_test=strat_test_set["MEDV"].copy()
X_test_prepared=my_pipeline.transform(X_test)
final_predictions=model.predict(X_test_prepared)
final_mse=mean_squared_error(Y_test,final_predictions)
final_rmse=np.sqrt(final_mse)
print(final_predictions,list(Y_test))
final_rmse
