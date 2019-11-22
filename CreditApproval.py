#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.simplefilter('ignore')

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import numpy as np
import pandas as pd


# ## 1. Data Processing and Preparation

# In[2]:


credit = pd.read_csv('Resources/credit-approval_csv - pandas.csv')
credit


# In[3]:


#get summary of numeric columns
credit.describe()


# In[4]:


# Replace "?" with NaN
credit.replace('?', np.NaN, inplace = True)
# Convert Age to numeric
credit["Age"] = pd.to_numeric(credit["Age"])
# credit_copy = credit[:,:]
#credit_copy = credit.copy()


# In[5]:


#replace missing values with mean values of numeric columns
credit.fillna(credit.mean(), inplace=True)


# In[6]:


def imputeWithMode(df):
    """ 
    Going through each columns and checking the type is object
    if it is object, impute it with most frequent value
    """
    for col in df:
        if df[col].dtypes == 'object':
            df[col] = df[col].fillna(df[col].mode().iloc[0])
imputeWithMode(credit)


# In[7]:


credit_drop=credit


# In[17]:


credit_drop=credit.drop(["ZipCode"],axis=1)


# In[46]:


credit_drop


# In[19]:


credit_drop.describe


# In[20]:


#LabelEncoder
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
# # Looping for each object type column
#Using label encoder to convert into numeric types
for col in credit_drop:
    if credit_drop[col].dtypes=='object':
        credit_drop[col]=LE.fit_transform(credit_drop[col])


# In[21]:


credit_drop.head()


# In[47]:


#HOT ENCODER
#convert to categorical data to dummy data
credit_dummies = pd.get_dummies(credit_drop, columns=[ "Married","EducationLevel", "Citizen", "DriversLicense", "Ethnicity"])
credit_dummies.head()


# In[48]:


credit_dummies.columns


# In[12]:


#credit_dummies=credit_drop


# In[49]:


credit_dummies.info()


# In[50]:


credit_dummies.describe()


# In[51]:


def plotDistPlot(col):
    """Flexibly plot a univariate distribution of observation"""
    sns.distplot(col)
    plt.show()
plotDistPlot(credit['Age'])
plotDistPlot(credit['Debt'])
plotDistPlot(credit['YearsEmployed'])
plotDistPlot(credit['CreditScore'])
plotDistPlot(credit['Income'])


# In[ ]:


#correlation matrix
corr = credit.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)


# In[ ]:


#scatterplot
sns.set()
cols = ['Age', 'Debt', 'BankCustomer','YearsEmployed','PriorDefault','CreditScore','Income']
sns.pairplot(credit[cols], size = 2.5)
plt.show();


# # 2. Data Modelling 

# # Random Forest

# In[52]:


credit_dummies.columns


# In[53]:


from sklearn.model_selection import train_test_split
# remove irrelevant features


# In[ ]:


#credit_drop=credit.drop(['DriversLicense', 'ZipCode'], axis=1)


# In[54]:


credit_dummies.to_numpy


# In[56]:


X,y = credit_dummies.iloc[:,0:40] , credit_dummies.iloc[:,40]

# Spliting the data into training and testing sets
X_train, X_test, y_train, Y_test = train_test_split(X,
                                y,
                                test_size=0.2,
                                random_state=123)


# In[57]:


X_train.head()


# In[58]:


#Scaling the data
from sklearn.preprocessing import MinMaxScaler

# Scaling X_train and X_test
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.transform(X_test)


# In[59]:


rescaledX = scaler.transform(X)


# In[60]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
rf = RandomForestClassifier(n_estimators=500)
rf.fit(rescaledX_train, y_train)
y_pred = rf.predict(rescaledX_test)
print("Random Forest classifier has accuracy of: ", rf.score(rescaledX_test, Y_test))
# Evaluate the confusion_matrix
confusion_matrix(Y_test, y_pred)


# In[61]:


importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()


# In[67]:


credit_dummies = credit_dummies.drop(['Approved'], axis=1)
features = credit_dummies.columns
importances = rf.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
plt.figure(figsize=(700,100))
plt.savefig('features.jpg')


# # Decision Tree

# In[68]:


#decision trees
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


# In[69]:


X_train.columns


# In[70]:


#model accuracy
print("Accuracy:", metrics.accuracy_score(Y_test, y_pred))


# In[ ]:


#graphing the decision tree
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import display, Image
import pydotplus

feature_cols = ['Gender','Age','Debt','Married','BankCustomer','EducationLevel','Ethnicity','YearsEmployed','PriorDefault','Employed','CreditScore','Citizen','Income','DriversLicense', 'ZipCode']

dot_data = StringIO()
export_graphviz(clf, out_file = dot_data,
                filled=True, rounded=True,
               special_characters=True, feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.gratph_from_dot_data(dot_data.getvalue())
graph.write_png('credit.png')
Image(graph.create_png())


# In[ ]:




