#!/usr/bin/env python
# coding: utf-8

# ## Market Basket Analysis using Apriori Algorithm in Python

# ### Importing Libraries

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ### Reading the Dataset

# In[5]:


df = pd.read_csv("Market Basket Analysis.csv")


# In[6]:


df


# ### Data Exploration

# In[7]:


x = df["itemDescription"].value_counts().sort_values(ascending=False)[:10]


# In[8]:


x


# In[9]:


plt.figure(figsize=(15,10))
sns.barplot(x.index,x.values)


# ### Apriori Algorithm

# Apriori algorithm is used for frequent itemset mining and association rule learning over relational databases. It proceeds by identifying the frequent individual items in the database and extending them to larger and larger item sets as long as those item sets appear sufficiently often in the database.
# 
# Support : It is the frequency of item a or combination of item A and B.
# 
# Confidence: It tells us how often the items a and b occur given that a is bought.
# 
# Lift: It tells us the strength of the rule.
# 
# Support = freq(A,B)/N
# 
# A and B Products
# 
# N is total Transactions
# 
# Confidence = freq(A,B) / freq(A)
# 
# Lift = Support / support(A) * Support(B)

# In[10]:


get_ipython().system('pip install mlxtend')


# In[11]:


from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# In[12]:


df


# In[13]:


df["Quantity"] = 1


# In[29]:


df


# In[37]:


transactions = df.groupby(['Member_number','itemDescription'])['Quantity'].sum().unstack().reset_index().set_index('Member_number')


# In[38]:


transactions = transactions.fillna(0)


# In[39]:


transactions


# In[40]:


def encode(x):
    if x<=0:
        return 0
    elif x>=0:
        return 1
    
basket = transactions.applymap(encode)


# In[41]:


basket


# In[42]:


from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
frequent_itemset = apriori(basket,min_support=0.06,use_colnames=True)
rules = association_rules(frequent_itemset,metric='lift',min_threshold=1)


# In[44]:


rules.head()


# In[50]:


rules[(rules['confidence']>0.4) & (rules['lift']>1)]

