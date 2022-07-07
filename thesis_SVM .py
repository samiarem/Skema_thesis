#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
import scipy.stats as statsµ
import statsmodels.api as sm
from statsmodels.formula.api import ols


# In[2]:


df = pd.read_csv('cancer.csv')


# In[3]:


df


# In[4]:


df.groupby('GENDER').size().plot(kind='pie', autopct='%.2f');


# 47 % of the dataset are females and 52% are males ==> dataset is balanced if we base it on gn


# In[5]:



# easier for prediction 


# In[ ]:





# In[6]:


df.groupby('LUNG_CANCER').size().plot(kind='pie', autopct='%.2f');
# in this dataset, most people have lung_cancer ( 87.38% ) which means that its not balanced


# WE ARE GONNA BALANCE THE DATASET 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[7]:


table = pd.pivot_table(df, values='SMOKING',  columns='GENDER',
               aggfunc='count')
table

# in case we see that males tend to have lung cancer more than females, it is not because they usually
#smoke more


# In[8]:


table1 = pd.pivot_table(df, values='ANXIETY',  columns='GENDER',
               aggfunc='count')
table1


# In[9]:


df.groupby('GENDER').AGE.hist();


# In[10]:


df.AGE.hist();


# In[11]:


table3 = pd.pivot_table(df, values='CHRONIC DISEASE',  columns='GENDER',
               aggfunc='count')
table3


# In[12]:


table4 = pd.pivot_table(df, values='PEER_PRESSURE',  columns='GENDER',
               aggfunc='count')
table4


# In[13]:


table5 = pd.pivot_table(df, values='ALCOHOL CONSUMING',  columns='GENDER',
               aggfunc='count')
table5 


# In[14]:


table6 = pd.pivot_table(df, values='WHEEZING',  columns='GENDER',
               aggfunc='count')
table6


# In[15]:


df


# In[16]:


df['GENDER'].replace('F', 0, inplace = True)
df['GENDER'].replace('M', 1, inplace = True)
df['LUNG_CANCER'].replace('NO',0, inplace = True )
df['LUNG_CANCER'].replace('YES', 1, inplace = True)


# In[17]:


df


# In[18]:



# setup random seed
np.random.seed(42)
#make the data
X= df.drop('LUNG_CANCER', axis = 1)
y= df['LUNG_CANCER']


# In[19]:



from sklearn.model_selection import train_test_split

X_train , X_test ,y_train, y_test = train_test_split(X,y,test_size=0.3)

#initiate SVM
from sklearn.svm import SVC
clf = SVC(kernel = 'linear', random_state = 0)
clf.fit(X_train, y_train)
# Evaluate LinearSVC
print(clf.score (X_test, y_test))
print(clf.score (X_train, y_train))
print(clf.score (X,y))



# In[20]:


from sklearn.model_selection import cross_val_score
cross_val_score=cross_val_score(clf,X,y,cv=5)
cross_val_score


# In[21]:


y_preds = clf.predict(X_test)
y_preds_train = clf.predict(X_train)


# In[22]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_preds))
# since our dataset is not balanced, we have 2 for class 0 and 60 for class 1 , we will take into consideration the Macro avg


# In[25]:


from sklearn.metrics import classification_report
print(classification_report(y_train,y_preds_train))


# In[ ]:


from sklearn.metrics import accuracy_score, precision_score
print(accuracy_score(y_test, y_preds))
# we have class imbalance ; class 0 =  7 and class 1 = 86, so we take into consideration macro avg !


# In[24]:


from sklearn.metrics import  roc_curve
#fit the classifier
clf.fit(X_train,y_train)
#Make preds with probabilities
y_probs = clf.predict_proba(X_test)
y_probs[:10] , len(y_probs)
# first column is probability that the label is 0 and the other one is 1


# In[26]:


clf(probability=True)


# In[27]:


from sklearn.metrics import  roc_curve
r_probs = [0 for _ in range(len(y_test))]
clf_probs = clf.predict_proba(X)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[28]:


# create a function  for plotting roc curve (matplotlib is already imported)
from sklearn.metrics import  roc_curve
def plot_roc_curve(fpr ,tpr):
    '''
     plots a roc curve given the false positive rate and true positive rate of a model
    '''
    #plot roc curve
    plt.plot(fpr,tpr,color='orange',label='ROC')
    # plot line with no predictive power(baseline)
    plt.plot([0,1],[0,1], color='darkblue', linestyle='--',label='Guessing')
    #customize plot
    plt.xlabel('False positive rate (fpr)')
    plt.ylabel('True positive rate (tpr)')
    plt.title('Receiver operating charatesrestics (ROC) curve')
    plt.legend()
    plt.show
plot_roc_curve(fpr,tpr)


# In[29]:


# make our confusion matrix more visual with  searborn.heatmap thats is built with matplotlib
import seaborn as sns
from sklearn.metrics import  confusion_matrix
# set the front scale
sns.set(font_scale=1.5)
# create a confusion matrix
conf_mat = confusion_matrix(y_test,y_preds)
# plot it using seaborn
sns.heatmap(conf_mat);


# In[30]:


from imblearn.over_sampling import SMOTE


# https://machinelearningmastery.com/combine-oversampling-and-undersampling-for-imbalanced-classification/#:~:text=Oversampling%20methods%20duplicate%20or%20create,of%20methods%20are%20used%20together.

# In[ ]:





# In[ ]:





# In[31]:


from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


# In[32]:


oversample = SMOTE()
undersample = RandomUnderSampler()
steps = [("o",oversample),("u",undersample)]
pipeline= Pipeline(steps=steps)
X , y = pipeline.fit_resample(X, y)


# In[33]:


y.groupby(y).size().plot(kind='pie', autopct='%.2f');


# In[34]:



from sklearn.model_selection import train_test_split

X_train , X_test ,y_train, y_test = train_test_split(X,y,test_size=0.3)

#initiate SVM
from sklearn.svm import SVC
clf = SVC(kernel = 'linear', random_state = 0)
clf.fit(X_train, y_train)
# Evaluate LinearSVC
print(clf.score (X_test, y_test))
print(clf.score (X_train, y_train))
print(clf.score (X,y))



# In[35]:


y_preds = clf.predict(X_test)
y_preds_train = clf.predict(X_train)


# In[36]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_preds))
print(classification_report(y_train,y_preds_train))

#oversampling is a good way to handle class imbalances, but it is just duplicating class 0, and it made the model overfit


# In[37]:


# make our confusion matrix more visual with  searborn.heatmap thats is built with matplotlib
import seaborn as sns
from sklearn.metrics import  confusion_matrix
# set the front scale
sns.set(font_scale=1.5)
# create a confusion matrix
conf_mat = confusion_matrix(y_test,y_preds)
# plot it using seaborn
sns.heatmap(conf_mat);


# let's try with undersampling

# In[ ]:





# In[ ]:




