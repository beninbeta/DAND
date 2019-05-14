#!/usr/bin/env python
# coding: utf-8

# ### Multiple Linear Regression Introduction
# 
# In this notebook (and following quizzes), you will be creating a few simple linear regression models, as well as a multiple linear regression model, to predict home value.
# 
# Let's get started by importing the necessary libraries and reading in the data you will be using.

# In[2]:


import numpy as np
import pandas as pd
import statsmodels.api as sm;

df = pd.read_csv('./house_prices.csv')
df.head()


# `1.` Using statsmodels, fit three individual simple linear regression models to predict price.  You should have a model that uses **area**, another using **bedrooms**, and a final one using **bathrooms**.  You will also want to use an intercept in each of your three models.
# 
# Use the results from each of your models to answer the first two quiz questions below.

# In[4]:


df['intercept'] = 1
y = df['price']
x = df[['intercept','area']]
area_mod = sm.OLS(y,x)
area_res = area_mod.fit()
area_res.summary()


# In[7]:


x = df[['intercept','bedrooms']]
bed_mod = sm.OLS(y,x)
bed_res = bed_mod.fit()
bed_res.summary()


# In[8]:


x = df[['intercept','bathrooms']]
bath_mod = sm.OLS(y,x)
bath_res = bath_mod.fit()
bath_res.summary()


# `2.` Now that you have looked at the results from the simple linear regression models, let's try a multiple linear regression model using all three of these variables  at the same time.  You will still want an intercept in this model.

# In[9]:


x= df[['intercept','area','bedrooms','bathrooms']]
mult_mod = sm.OLS(y,x)
mult_res = mult_mod.fit()
mult_res.summary()


# `3.` Along with using the **area**, **bedrooms**, and **bathrooms** you might also want to use **style** to predict the price.  Try adding this to your multiple linear regression model.  What happens?  Use the final quiz below to provide your answer.

# In[10]:


x= df[['intercept','area','bedrooms','bathrooms','style']]
mult_mod = sm.OLS(y,x)
mult_res = mult_mod.fit()
mult_res.summary()


# In[ ]:




