#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as pt
import warnings
warnings.filterwarnings('ignore')


# In[ ]:





# In[9]:


#Amazon data
df1=pd.read_excel('amz_com-ecommerce_sample.xlsx')
df1.head()


# In[10]:


df1.info()


# In[11]:


df1.columns


# In[13]:


#Drop columns
df1.drop(['uniq_id','crawl_timestamp', 'product_url','product_category_tree',
            'image', 'is_FK_Advantage_product', 'description', 'product_rating',
       'overall_rating', 'brand', 'product_specifications'],axis=1,inplace=True)


# In[14]:


df1.head()


# In[15]:


df1.columns


# In[16]:


df1.shape


# In[ ]:





# In[22]:


#flipkart data
df2=pd.read_excel('flipkart_com-ecommerce_sample.xlsx')
df2.head()


# In[23]:


#Drop columns
df2.drop(['uniq_id','crawl_timestamp', 'product_url','product_category_tree',
            'image', 'is_FK_Advantage_product', 'description', 'product_rating',
       'overall_rating', 'brand', 'product_specifications'],axis=1,inplace=True)


# In[24]:


df2.head()


# In[25]:


df2.columns


# In[27]:


df2.shape


# In[ ]:





# In[32]:


#Merge both data
df= pd.merge(df2, df1, on="pid")


# In[33]:


df.head()


# In[35]:


df.drop('pid',axis=1,inplace=True)


# In[36]:


df.head()


# In[39]:


df.rename(columns={'product_name_x': 'Product name in Flipcart',
                    'retail_price_x': 'Retail price in Flipcart',
                     'discounted_price_x': 'Discounted price in Flipcart',
                    'product_name_y': 'Product name in Amazon',
                    'retail_price_y': 'Retail price in Amazon',
                     'discounted_price_y': 'Discounted price in Amazon',})


# In[ ]:




