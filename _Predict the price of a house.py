#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data=pd.read_excel('DS - Assignment Part 1 data set.xlsx')


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


data.corr()


# In[8]:


#Checking missing values
data.isnull().sum()


# #### No missing values in this dataset

# In[9]:


data=data.drop("Transaction date",axis=1)


# In[10]:


data.head()


# In[ ]:





# In[39]:


#Relation between dependent and independent features using scatter plot
dt=data.copy()
for feature in data.columns:
    plt.scatter(data[feature],data['House price of unit area'])
    plt.xlabel(feature)
    plt.ylabel('House price of unit area')
    plt.show()


# In[ ]:





# In[11]:


x=data.iloc[:,:-1]
y=data.iloc[:,-1]


# In[12]:


x.head()


# In[13]:


y.head()


# In[14]:


#Train test split


# In[15]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42)


# In[16]:


print(x_train.shape)
x_train.head()


# In[17]:


print(x_test.shape)
x_test.head()


# In[ ]:





# In[18]:


import seaborn as sns
sns.pairplot(x_train)


# In[ ]:





# In[19]:


#Linear Regression
from sklearn.linear_model import LinearRegression
linear_regressor=LinearRegression()
linear_regressor.fit(x_train,y_train)


# In[20]:


print(linear_regressor.score(x_train,y_train))
print(linear_regressor.score(x_test,y_test))


# #### Here we can see that difference between train and test dataset is 0.006284545574150924

# In[ ]:





# In[68]:


#Decision tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
decision_tree=DecisionTreeRegressor()


# In[71]:


parameters={'max_depth' : [3,5,7,9], 'min_samples_split':[2,4,6],'min_samples_leaf':[1,2,3]}
Decision_tree_reg=GridSearchCV(decision_tree,parameters,scoring='neg_mean_squared_error',cv=5)


# In[72]:


Decision_tree_reg.fit(x_train,y_train)


# In[74]:


print(Decision_tree_reg.best_params_)
print(Decision_tree_reg.best_score_)


# In[77]:


decision_tree=DecisionTreeRegressor(max_depth=3,min_samples_leaf=1,min_samples_split=2)
decision_tree.fit(x_train,y_train)


# In[78]:


print(decision_tree.score(x_train,y_train))
print(decision_tree.score(x_test,y_test))


# #### Here we can see that difference between train and test dataset is 0.025053479384892996

# In[ ]:





# In[23]:


#Random forest
from sklearn.ensemble import RandomForestRegressor


# In[81]:


RandomForest = RandomForestRegressor()


# In[84]:


parameters={'max_depth' : [3,5,7],'n_estimators':[100,500,1000]}
Randomforest_reg=GridSearchCV(RandomForest,parameters,scoring='neg_mean_squared_error',cv=5)


# In[85]:


Randomforest_reg.fit(x_train,y_train)


# In[86]:


print(xgboost_reg.best_params_)
print(xgboost_reg.best_score_)


# In[116]:


RandomForest = RandomForestRegressor(n_estimators=100,max_depth=3,criterion='mse',min_samples_leaf=4,min_samples_split=2,random_state=0)
RandomForest.fit(x_train,y_train)


# In[127]:


print(RandomForest.score(x_train,y_train))
print(RandomForest.score(x_test,y_test))


# #### Here we can see that difference between train and test dataset is 0.024878469104981038

# In[ ]:





# In[ ]:





# In[26]:


#xgboost
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn import metrics


# In[27]:


xg_reg = xgb.XGBRegressor()


# In[28]:


parameters={'max_depth' : [3,5,7,9],'n_estimators':[100,500,1000]}
xgboost_reg=GridSearchCV(xg_reg,parameters,scoring='neg_mean_squared_error',cv=5)


# In[29]:


xgboost_reg


# In[30]:


xgboost_reg.fit(x_train,y_train)


# In[31]:


print(xgboost_reg.best_params_)
print(xgboost_reg.best_score_)


# In[ ]:





# In[124]:


xg_reg = xgb.XGBRegressor(max_depth= 3, n_estimators= 100)
xg_reg.fit(x_train,y_train)


# In[125]:


print(xg_reg.score(x_train,y_train))
print(xg_reg.score(x_test,y_test))


# In[34]:


expected_y  = y_test
predicted_y = xg_reg.predict(x_test)


# In[35]:


print(metrics.r2_score(expected_y, predicted_y))
print(metrics.mean_squared_log_error(expected_y, predicted_y))


# #### Here we can see that difference between train and test dataset is 0.30344038907899773

# In[ ]:


# So we can take Randomforest model for prediction


# In[134]:


predicted_y=RandomForest.predict(x_test)
predicted_y


# In[135]:


print(metrics.r2_score(expected_y, predicted_y))
print(metrics.mean_squared_log_error(expected_y, predicted_y))


# In[ ]:




