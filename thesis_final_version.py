#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('cancer.csv')
df


# In[3]:


df.groupby('GENDER').size().plot(kind='pie', autopct='%.2f');
## 47 % of the dataset are females and 52% are males ==> dataset is balanced if we base it on gn


# In[4]:


df.groupby('LUNG_CANCER').size().plot(kind='pie', autopct='%.2f');
# in this dataset, most people have lung_cancer ( 87.38% ) which means that its not balanced


# In[5]:


table = pd.pivot_table(df, values='SMOKING',  columns='GENDER',
               aggfunc='count')
table
## in case we see that males tend to have lung cancer more than females, it is not because they usually smoke more


# In[6]:


table1 = pd.pivot_table(df, values='ANXIETY',  columns='GENDER',
               aggfunc='count')
table1


# In[7]:


df.groupby('GENDER').AGE.hist();


# In[8]:


df.AGE.hist();


# In[9]:


table3 = pd.pivot_table(df, values='CHRONIC DISEASE',  columns='GENDER',
               aggfunc='count')
table3


# In[10]:


table4 = pd.pivot_table(df, values='PEER_PRESSURE',  columns='GENDER',
               aggfunc='count')
table4


# In[11]:


table5 = pd.pivot_table(df, values='ALCOHOL CONSUMING',  columns='GENDER',
               aggfunc='count')
table5 


# In[12]:


table6 = pd.pivot_table(df, values='WHEEZING',  columns='GENDER',
               aggfunc='count')
table6


# In[ ]:





# In[13]:


df['GENDER'].replace('F', 0, inplace = True)
df['GENDER'].replace('M', 1, inplace = True)
df['LUNG_CANCER'].replace('NO',0, inplace = True )
df['LUNG_CANCER'].replace('YES', 1, inplace = True)


# In[14]:


df


# ### SVM
# 

# In[15]:


# setup random seed
np.random.seed(42)
#make the data
X= df.drop('LUNG_CANCER', axis = 1)
y= df['LUNG_CANCER']


# In[16]:


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


# In[21]:


from sklearn.model_selection import cross_val_score
cross_val_score=cross_val_score(clf,X,y,cv=5)
cross_val_score


# In[22]:


y_preds = clf.predict(X_test)
y_preds_train = clf.predict(X_train)


# In[60]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_preds))
# since our dataset is not balanced, we have 2 for class 0 and 60 for class 1 , we will take into consideration the Macro avg


# In[61]:


from sklearn.metrics import classification_report
print(classification_report(y_train,y_preds_train))


# In[62]:



from sklearn.metrics import accuracy_score, precision_score
print(accuracy_score(y_test, y_preds))
# we have class imbalance ; class 0 =  7 and class 1 = 86, so we take into consideration macro avg !


# In[24]:


from sklearn.metrics import accuracy_score, precision_score
print(accuracy_score(y_train, y_preds_train))


# In[63]:


from sklearn.metrics import  roc_curve
#fit the classifier
clf.fit(X_train,y_train)
#Make preds with probabilities
y_probs = clf.predict_proba(X_test)
y_probs[:10] , len(y_probs)
# first column is probability that the label is 0 and the other one is 1
# trying to do ROC curve


# In[64]:


from sklearn.metrics import  roc_curve
r_probs = [0 for _ in range(len(y_test))]
clf_probs = clf.predict_proba(X)
#trying to do ROC curve


# In[65]:


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
#Trying to do ROC curve


# In[66]:


# make our confusion matrix more visual with  searborn.heatmap thats is built with matplotlib
import seaborn as sns
from sklearn.metrics import  confusion_matrix
# set the front scale
sns.set(font_scale=1.5)
# create a confusion matrix
conf_mat = confusion_matrix(y_test,y_preds)
# plot it using seaborn
sns.heatmap(conf_mat);


# In[67]:



pip install imbalanced-learn


# In[68]:


from imblearn.over_sampling import SMOTE


# In[69]:


from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


# In[70]:


oversample = SMOTE()
undersample = RandomUnderSampler()
steps = [("o",oversample),("u",undersample)]
pipeline= Pipeline(steps=steps)
X , y = pipeline.fit_resample(X, y)


# In[71]:


y.groupby(y).size().plot(kind='pie', autopct='%.2f');


# In[72]:


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


# In[73]:


y_preds = clf.predict(X_test)
y_preds_train = clf.predict(X_train)


# In[74]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_preds))
print(classification_report(y_train,y_preds_train))
#oversampling is a good way to handle class imbalances, but it is just duplicating class 0, and it made the model overfit


# ### RandomForestClassifier

# In[75]:


from sklearn.ensemble import RandomForestClassifier


from sklearn.model_selection import train_test_split

X_train , X_test ,y_train, y_test = train_test_split(X,y,test_size=0.3)

clf= RandomForestClassifier(n_estimators=100)
clf = clf.fit(X, y)
# Evaluatethe model
print(clf.score (X_test, y_test))
print(clf.score (X_train, y_train))
print(clf.score (X,y))


# In[76]:


from sklearn.model_selection import cross_val_score
cross_val_score=cross_val_score(clf,X,y,cv=5)
cross_val_score


# In[77]:


y_preds = clf.predict(X_test)
y_preds_train = clf.predict(X_train)
y_preds


# In[78]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_preds))


# In[79]:


print(classification_report(y_train,y_preds_train))


# In[80]:


clf.get_params()


# In[81]:



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


# In[82]:


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


# In[83]:


rs_clf.best_params_ #tells you best combination


# In[84]:


rs_y_preds= rs_clf.predict(X_test)


# In[85]:


print(rs_clf.score (X_test, y_test))
print(rs_clf.score (X_train, y_train))
print(rs_clf.score (X,y))


# In[86]:


print(classification_report(y_test,rs_y_preds))


# In[87]:



# since its a lot of combination, its gonna take a lot of time and requires very strong computer, so we gonna base our grid to the best parameters that were chosen by
#randomized searchCV
grid_2= {'n_estimators': [ 50, 100, 200],
            'max_depth': [None,10,20],
            'max_features': ['auto', 'sqrt'],
            'min_samples_split': [6],
            'min_samples_leaf': [1, 2]}
#as you can see, compared to the previous grid, we reduced it and we have less parameters


# In[88]:


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


# In[90]:


gs_clf.best_params_


# In[91]:


gs_y_preds =  gs_clf.predict(X_test)


# In[92]:


print(classification_report(y_test,gs_y_preds)) 
#recall decreased significantly


# ### Decision Tree

# In[93]:



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


# In[94]:


from sklearn.model_selection import cross_val_score
cross_val_score=cross_val_score(clf,X,y,cv=5)
cross_val_score


# In[95]:



y_preds = clf.predict(X_test)
y_preds_train = clf.predict(X_train)
y_preds


# In[96]:



from sklearn.metrics import classification_report
print(classification_report(y_test,y_preds))


# In[97]:


print(classification_report(y_train,y_preds_train))


# ## AdaBoost

# In[98]:


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


# In[99]:


from sklearn.model_selection import cross_val_score
cross_val_score=cross_val_score(clf,X,y,cv=5)
cross_val_score


# In[100]:


y_preds = clf.predict(X_test)
y_preds_train = clf.predict(X_train)
y_preds


# In[101]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_preds))


# In[102]:


print(classification_report(y_train,y_preds_train))


# In[ ]:





# In[ ]:




