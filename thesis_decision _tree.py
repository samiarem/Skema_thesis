#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:



df = pd.read_csv('cancer.csv')


# In[5]:


df['GENDER'].replace('F', 0, inplace = True)
df['GENDER'].replace('M', 1, inplace = True)
df['LUNG_CANCER'].replace('NO',0, inplace = True )
df['LUNG_CANCER'].replace('YES', 1, inplace = True)


# In[6]:


df


# In[15]:


plt.figure(figsize=(15,8))
sns.heatmap(df.corr(), annot=True)


# In[7]:



# setup random seed
np.random.seed(42)
#make the data
X= df.drop('LUNG_CANCER', axis = 1)
y= df['LUNG_CANCER']


# In[ ]:





# In[10]:



from sklearn.model_selection import train_test_split

X_train , X_test ,y_train, y_test = train_test_split(X,y,test_size=0.3)

#initiate decision_tree
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)
# Evaluate LinearSVC
print(clf.score (X_test, y_test))
print(clf.score (X_train, y_train))
print(clf.score (X,y))


# In[11]:


from sklearn.model_selection import cross_val_score
cross_val_score=cross_val_score(clf,X,y,cv=5)
cross_val_score


# In[14]:


y_preds = clf.predict(X_test)
y_preds_train = clf.predict(X_train)
y_preds


# In[13]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_preds))


# In[17]:



print(classification_report(y_train,y_preds_train))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[20]:


from sklearn.metrics import  roc_curve
#fit the classifier
clf.fit(X_train,y_train)
#Make preds with probabilities
y_probs = clf.predict_proba(X_test)
y_probs[:10] , len(y_probs)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[23]:


fpr, tpr, thresholds =roc_curve(y_test,y_probs)


# In[ ]:




