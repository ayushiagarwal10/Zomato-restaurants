
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[6]:


data = pd.read_csv("zomato.csv")


# In[7]:


data.info()


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


data.apply(lambda x : sum(x.isnull()))


# In[ ]:


sns.countplot(x=data['listed_in(type)'],hue =data['online_order'])


# In[ ]:


sns.countplot(x=data['rest_type'],hue =data['online_order'])


# In[ ]:


p1 = data['cuisines'].value_counts()
print(p1)


# In[ ]:


data["votes"].plot(kind="hist", figsize=(9,9))


# In[ ]:


numerical = data.dtypes[data.dtypes =="object"].index

print(numerical)


# In[ ]:


data['rest_type'].value_counts()
sns.countplot(data['rest_type'])


# In[ ]:


X = data.drop(['approx_cost(for two people)'],axis=1).values


# In[ ]:


Y = data['approx_cost(for two people)'].values


# In[ ]:


X_train,Y_train,X_test,Y_test =train_test_split(X, Y ,test_size =0.25,random_state=34)


# In[ ]:


data['menu_item'].value_counts()


# In[ ]:


(data['rest_type']=='Bar').value_counts()


# In[8]:


delivery_drop = data[data['rest_type']=='Delivery'].sort_values(by='approx_cost(for two people)', ascending=True)[['name','approx_cost(for two people)','menu_item','votes','reviews_list','rate','url', 'address', 'phone',
       'location', 'rest_type', 'dish_liked', 'cuisines',
       'listed_in(type)', 'listed_in(city)']].head(697)


# In[ ]:


delivery_drop.head()


# In[ ]:


delivery_drop.shape


# In[ ]:


data.info()


# In[ ]:


data.apply(lambda x :sum(x.isnull()))


# In[ ]:


sns.countplot(x=data['listed_in(type)'],hue =data['online_order'])


# In[ ]:


delivery_drop.info()


# In[ ]:


delivery_drop.shape


# In[ ]:


(data['rest_type']=='Bar').value_counts()


# In[ ]:


bar_drop = data[data['rest_type']=='Bar'].sort_values(by='approx_cost(for two people)', ascending=True)[['name','approx_cost(for two people)','menu_item','votes','reviews_list','rate','url', 'address', 'phone',
       'location', 'rest_type', 'dish_liked', 'cuisines','online_order',
       'listed_in(type)', 'listed_in(city)']].head(697)


# In[ ]:


bar_drop.head()


# In[ ]:


bar_drop.shape


# In[ ]:


(data['rest_type']=='Pub').value_counts()


# In[ ]:


pub_drop = data[data['rest_type']=='Pub'].sort_values(by='approx_cost(for two people)', ascending=True)[['name','approx_cost(for two people)','menu_item','votes','reviews_list','rate','url', 'address', 'phone',
       'location', 'rest_type', 'dish_liked', 'cuisines','online_order',
       'listed_in(type)', 'listed_in(city)']].head(357)


# In[ ]:


pub_drop.head()


# In[ ]:


pub_drop.shape


# In[ ]:


delivery_drop['approx_cost(for two people)'].value_counts().sort_index().plot.bar()


# In[ ]:


delivery_drop['rate'].value_counts().sort_index().plot.bar()


# In[ ]:


delivery_drop['rate'].value_counts()


# In[ ]:


delivery_drop['approx_cost(for two people)'].value_counts()


# In[ ]:


delivery_drop[delivery_drop['approx_cost(for two people)'] < 200]['approx_cost(for two people)'].plot.line()


# In[9]:


delivery_drop.plot.bar(stacked=True)

