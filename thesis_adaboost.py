#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:



df = pd.read_csv('cancer.csv')


# In[3]:


df['GENDER'].replace('F', 0, inplace = True)
df['GENDER'].replace('M', 1, inplace = True)
df['LUNG_CANCER'].replace('NO',0, inplace = True )
df['LUNG_CANCER'].replace('YES', 1, inplace = True)


# In[4]:


df


# In[5]:



# setup random seed
np.random.seed(42)
#make the data
X= df.drop('LUNG_CANCER', axis = 1)
y= df['LUNG_CANCER']


# In[9]:


from sklearn.model_selection import train_test_split

X_train , X_test ,y_train, y_test = train_test_split(X,y,test_size=0.3)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
clf = AdaBoostClassifier(n_estimators=100, random_state=0)
clf.fit(X, y)
AdaBoostClassifier(n_estimators=100, random_state=0)

print(clf.score (X_test, y_test))
print(clf.score (X_train, y_train))
print(clf.score (X,y))



# In[10]:


from sklearn.model_selection import cross_val_score
cross_val_score=cross_val_score(clf,X,y,cv=5)
cross_val_score


# In[11]:


y_preds = clf.predict(X_test)
y_preds_train = clf.predict(X_train)
y_preds


# In[12]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_preds))


# In[13]:



print(classification_report(y_train,y_preds_train))


# In[14]:


clf.get_params()


# In[20]:


from sklearn.ensemble import AdaBoostClassifier
np.random.seed(42)
#shuffle the data
df_shuffled= df.sample(frac=1)
# split into X and y
X= df_shuffled.drop('LUNG_CANCER', axis = 1)
y= df_shuffled['LUNG_CANCER']
# split the data into train, validation and test sets
train_split = round(0.7*len(df_shuffled)) #70% of data
valid_split = round(train_split +0.15*len(df_shuffled )) #15% of data
X_train,  y_train = X[:train_split], y[:train_split]
X_valid, y_valid = X[train_split:valid_split], y[train_split:valid_split]
X_test , y_test = X[valid_split:], y[valid_split:]
len(X_train), len(X_valid), len(X_test)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
# Make baseline predictions
y_preds = clf.predict(X_valid)


# In[26]:


from sklearn.model_selection import RandomizedSearchCV
grid = {'n_estimators' :[10,100,200,500,1000,1200], #numbers are based on researches
       "base_estimator":[None,5,10,20,30],
       'learning_rate':[1,2,3,4],
       'random_state': [0,1,2,4,6],
        'algorithm': ['SAMME']}
np.random.seed(42)
#split into X and y
X= df_shuffled.drop('LUNG_CANCER', axis = 1)
y= df_shuffled['LUNG_CANCER']
# split into train and test set
X_test, X_train , y_test,y_train = train_test_split(X,y, test_size=0.3)
# Initiate RandomForestClassifier
clf = AdaBoostClassifier(n_estimators=100, random_state=0) #n_jobs choose how much computer cpu dedicate  to ml model
# setup randomized searched cv
rs_clf= RandomizedSearchCV(estimator=clf,
                          param_distributions=grid,
                          n_iter=10, #number of models to try
                          cv=5, # means cross validation
                          verbose=2) 
# fit the randomizedsearchCV version of clf
rs_clf.fit(X_train, y_train);
#this is used to try different combination of parameters and figure out the best combination of parameters to maximise accuracy of the Ml MODEL


# In[ ]:




