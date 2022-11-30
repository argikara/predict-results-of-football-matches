#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn import preprocessing
#import pymrmr
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import SelectKBest
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from mrmr import mrmr_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, SelectFromModel, chi2, mutual_info_classif
from sklearn.svm import SVC
import math
from imblearn.over_sampling import SVMSMOTE
from sklearn.model_selection import LeaveOneOut


# In[2]:


#load the dataset from preprocessing
df = pd.read_csv('df.csv')


# In[3]:


#drop the first 5 fixtures of every year
df.drop(df.index[0:50], inplace=True)


# In[4]:


df = df.reset_index(drop=True)


# In[5]:


df.drop(df.index[330:380], inplace=True)


# In[6]:


df = df.reset_index(drop=True)


# In[7]:


df.drop(df.index[660:710], inplace=True)


# In[8]:


df = df.reset_index(drop=True)


# In[9]:


df.drop(df.index[990:1040], inplace=True)


# In[10]:


df = df.reset_index(drop=True)


# In[11]:


df.drop(df.index[1320:1370], inplace=True)


# In[12]:


df = df.reset_index(drop=True)


# In[13]:


df.drop(df.index[1650:1700], inplace=True)


# In[14]:


df = df.reset_index(drop=True)


# In[15]:


df = df.drop(['home_team','away_team', 'year'], axis=1)


# In[16]:


corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')


# In[17]:


df = df.drop(['home_avg_pas','away_avg_pas', 'home_team_home_points', 'away_team_away_points',
               'home_team_home_avg_goals', 'away_team_away_avg_goals'], axis=1)


# In[18]:


df['win'] = df['win'].astype(int)


# In[19]:


lab_enc = preprocessing.LabelEncoder()
encoded = lab_enc.fit_transform(df['win'])


# In[20]:


df['win'] = encoded


# In[21]:


def grid(df,clf,para):
    x = df.drop(['win'], axis=1)
    
    y = df['win']
    
    selected_features = mrmr_classif(x, y, K = 35)
    
    x_mrmr = df[selected_features]
    
    cv = LeaveOneOut()
    sm = SVMSMOTE()
    kbest = SelectKBest()
    
    classifier = clf 
    parameters = para
    
    
    x_train, x_test, y_train, y_test = train_test_split(x_mrmr, y, train_size=0.8, random_state=88)
    
    pipe= Pipeline(steps=[('sm',sm), ('kbest',kbest), ('classifier', classifier)])
    
    pipe.fit(x_train, y_train)
    
    cv= GridSearchCV(pipe, parameters, scoring='balanced_accuracy', verbose=1, n_jobs=-1, cv=cv)
    
    grid_result=cv.fit(x_train,y_train)
    
    
    return grid_result.best_score_, grid_result.best_params_


# In[42]:


clf = [(RandomForestClassifier(), {'classifier__class_weight':['balanced'],
                                                 'classifier__bootstrap': [True],
                                                 'classifier__max_depth': [5, 10, 20],
                                                 'classifier__max_features': ['auto'],
                                                 'classifier__min_samples_leaf': [1, 2, 3],
                                                 'classifier__min_samples_split': [2, 5, 10],
                                                 'classifier__n_estimators': [100, 200, 300]}),
       (KNeighborsClassifier(), {'classifier__n_neighbors':[1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
                                }),
        (GradientBoostingClassifier(), {'classifier__loss':['deviance', 'exponential'],
                                                  'classifier__n_estimators' : [30, 60, 100, 150, 200],
                                                  'classifier__learning_rate': [0.01, 0.05, 0.1, 0.15],
                                                  'classifier__max_depth': [1, 6, 10, 15, 20],
                                                  }),
        (SVC(), {'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                          'classifier__C': [1, 10, 100, 1000],
                          'classifier__class_weight':['balanced'],
                          'classifier__gamma' : [0.001, 0.01, 0.1, 1]})]


# In[24]:


output={}
for i in clf:
    print(i[0])
    bestscore,bestparams = grid(df,i[0],i[1])
    output.update({i[0]:[bestscore,bestparams]})    

