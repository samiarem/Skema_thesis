#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:



df = pd.read_csv('cancer.csv')


# In[4]:


df['GENDER'].replace('F', 0, inplace = True)
df['GENDER'].replace('M', 1, inplace = True)
df['LUNG_CANCER'].replace('NO',0, inplace = True )
df['LUNG_CANCER'].replace('YES', 1, inplace = True)


# In[5]:


df


# In[6]:



# setup random seed
np.random.seed(42)
#make the data
X= df.drop('LUNG_CANCER', axis = 1)
y= df['LUNG_CANCER']


# In[7]:


#choose the right model and hyperparameters
from sklearn.ensemble import RandomForestClassifier


from sklearn.model_selection import train_test_split

X_train , X_test ,y_train, y_test = train_test_split(X,y,test_size=0.3)

clf= RandomForestClassifier(n_estimators=100)
clf = clf.fit(X, y)
# Evaluatethe model
print(clf.score (X_test, y_test))
print(clf.score (X_train, y_train))
print(clf.score (X,y))


# In[8]:


from sklearn.model_selection import cross_val_score
cross_val_score=cross_val_score(clf,X,y,cv=5)
cross_val_score


# In[9]:


y_preds = clf.predict(X_test)
y_preds_train = clf.predict(X_train)
y_preds


# In[10]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_preds))


# In[11]:



print(classification_report(y_train,y_preds_train))


# In[12]:


clf.get_params()


# In[13]:


from sklearn.ensemble import RandomForestClassifier
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


# In[14]:


from sklearn.model_selection import RandomizedSearchCV
grid = {'n_estimators' :[10,100,200,500,1000,1200], #numbers are based on researches
       "max_depth":[None,5,10,20,30],
       'max_features':['auto','sqrt'],
       'min_samples_split': [2,4,6],
       'min_samples_leaf':[1,2,4]}
np.random.seed(42)
#split into X and y
X= df_shuffled.drop('LUNG_CANCER', axis = 1)
y= df_shuffled['LUNG_CANCER']
# split into train and test set
X_test, X_train , y_test,y_train = train_test_split(X,y, test_size=0.3)
# Initiate RandomForestClassifier
clf =RandomForestClassifier(n_jobs=None) #n_jobs choose how much computer cpu dedicate  to ml model
# setup randomized searched cv
rs_clf= RandomizedSearchCV(estimator=clf,
                          param_distributions=grid,
                          n_iter=10, #number of models to try
                          cv=5, # means cross validation
                          verbose=2) 
# fit the randomizedsearchCV version of clf
rs_clf.fit(X_train, y_train);
#this is used to try different combination of parameters and figure out the best combination of parameters to maximise accuracy of the Ml MODEL


# In[15]:


rs_clf.best_params_ #tells you best combination


# In[16]:


rs_y_preds= rs_clf.predict(X_test)


# In[19]:


print(rs_clf.score (X_test, y_test))
print(rs_clf.score (X_train, y_train))
print(rs_clf.score (X,y))


# In[17]:



print(classification_report(y_test,rs_y_preds))


# In[ ]:





# In[21]:


# since its a lot of combination, its gonna take a lot of time and requires very strong computer, so we gonna base our grid to the best parameters that were chosen by
#randomized searchCV
grid_2= {'n_estimators': [ 50, 100, 200],
            'max_depth': [None,10,20],
            'max_features': ['auto', 'sqrt'],
            'min_samples_split': [6],
            'min_samples_leaf': [1, 2]}
#as you can see, compared to the previous grid, we reduced it and we have less parameters


# In[22]:


from sklearn.model_selection import GridSearchCV, train_test_split
np.random.seed(42)
#split into X and y
X= df_shuffled.drop('LUNG_CANCER', axis = 1)
y= df_shuffled['LUNG_CANCER']
# split into train and test set
X_test, X_train , y_test,y_train = train_test_split(X,y, test_size=0.3)
# Initiate RandomForestClassifier
clf =RandomForestClassifier(n_jobs=None) #n_jobs choose how much computer cpu dedicate  to ml model
# setup GridSearchCV
gs_clf= GridSearchCV(estimator=clf,
                          param_grid=grid_2, 
                          cv=5, # means cross validation
                          verbose=2) 
# fit the GridsearchCV version of clf
gs_clf.fit(X_train, y_train); #gs means greadserch
#this is used to try different combination of parameters and figure out the best combination of parameters to maximise accuracy of the Ml MODEL


# In[23]:


gs_clf.best_params_


# In[24]:


gs_y_preds =  gs_clf.predict(X_test)


# In[25]:



print(classification_report(y_test,gs_y_preds)) 


# In[ ]:




