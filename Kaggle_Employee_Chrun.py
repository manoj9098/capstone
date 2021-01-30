#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# In[4]:


data = pd.read_csv('Dataset/HR_comma_sep.csv')


# In[5]:





# In[6]:





# In[7]:





# In[8]:





# In[9]:





# In[10]:





# In[11]:




# In[12]:





# In[13]:


X = data.drop(labels='left',axis=1).values
y = data['left'].values


# In[14]:


from sklearn.preprocessing import LabelEncoder
Lc_salary = LabelEncoder()
X[:,8] = Lc_salary.fit_transform(X[:,8])


# In[15]:




# In[16]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X[:,0:5] = sc.fit_transform(X[:,0:5])


# In[17]:




# In[18]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[7])], remainder = 'passthrough')
X = ct.fit_transform(X)


# In[19]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 42)


# In[20]:


from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)


# In[21]:


# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Model Precision
print("Precision:",metrics.precision_score(y_test, y_pred))
# Model Recall
print("Recall:",metrics.recall_score(y_test, y_pred))


# In[22]:


metrics.confusion_matrix(y_test, y_pred)


# In[23]:


import pickle


# In[24]:


pickle.dump(gb, open('model.pkl','wb'))
pickle.dump(Lc_salary, open('encoder.pkl','wb'))
pickle.dump(sc, open('scaler.pkl','wb'))
pickle.dump(ct, open('transformer.pkl','wb'))


# In[29]:




# In[30]:


np.count_nonzero(y_pred == 1)


# In[31]:


y_pred


# In[ ]:




