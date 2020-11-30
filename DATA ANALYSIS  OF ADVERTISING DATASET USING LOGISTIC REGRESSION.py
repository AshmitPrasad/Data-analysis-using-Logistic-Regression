#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame,Series


# In[6]:


df=pd.read_csv("advertising.csv")


# In[7]:


df.head()


# In[8]:


df.describe()


# In[9]:


df.info()


# # data analysis

# In[18]:


sns.set_style("darkgrid")
sns.distplot(df['Age'],kde=False,bins=30,hist_kws={"alpha":0.85})


# In[19]:


sns.jointplot(x="Age",y="Area Income",data=df,color="red")


# In[22]:


sns.jointplot(x="Age",y="Daily Time Spent on Site",color="green",kind="kde",data=df)


# In[24]:


sns.jointplot(x="Daily Time Spent on Site",y="Daily Internet Usage",data=df,color="black")


# In[28]:


sns.pairplot(df,hue="Clicked on Ad");


# # Logistic Regression

# In[29]:


from sklearn.model_selection import train_test_split


# In[30]:


X = df[["Daily Time Spent on Site", 'Age', 'Area Income','Daily Internet Usage', 'Male']]


# In[31]:


X_train, X_test, Y_train, Y_test = train_test_split(X,df["Clicked on Ad"],test_size=0.30,random_state=42)


# In[32]:


X_train, X_test, Y_train, Y_test = train_test_split(X,df["Clicked on Ad"],test_size=0.30,random_state=42)


# In[34]:


from sklearn.linear_model import LogisticRegression


# In[35]:


model = LogisticRegression()
model.fit(X_train,Y_train)


# # Predictions and Evaluations

# In[40]:


pred= model.predict(X_test)


# In[41]:


from sklearn.metrics import classification_report


# In[42]:


print(classification_report(Y_test,pred))


# In[ ]:




