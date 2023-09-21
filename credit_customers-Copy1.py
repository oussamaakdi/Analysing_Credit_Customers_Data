#!/usr/bin/env python
# coding: utf-8

# # Upload Dependencies

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import mglearn


# In[2]:


ddf = pd.read_csv("\credit_customers.csv")
pd.set_option('display.max_column',21)
df= ddf.copy()


# In[3]:


df.info()


# # Data Exploration

# In[4]:


ff=pd.get_dummies(df.drop("class",axis=1))


# In[5]:


df.describe()

Separating our dataset in two parts, each one gathers all samples belonging to one class
# In[6]:


good_data = df[df["class"]=='good']
bad_data = df[df["class"]=="bad"]

Categorical features COUNT PLOTS : 
Based on these plots we can't get accurate information about the distribution of the two classes because the proportions of the good and bad classes in our dataset are not equivalent
But in spite of that! the credit-history variable shows a nice separation of the two distributions
# In[7]:


categorical_features = df.select_dtypes('object').columns
n=len(categorical_features)
fig , axes = plt.subplots(14,1,figsize=(18,84))
for ax,feature in zip(axes.ravel(),categorical_features):
        sns.countplot(data=good_data,x=feature,ax=ax,alpha=0.5,color='grey',saturation=1,label="good_cases")
        sns.countplot(data=bad_data,x=feature,ax=ax,alpha=0.5,color='r',saturation=1,label="bad_cases")
axes.ravel()[0].legend(loc="best")

Potting the histograms of individual numerical features in our dataset :
for the same reason explained above we can't make class judgement basing on these plots
# In[8]:


numerical_features = df.select_dtypes('float64').columns
fig , axes =plt.subplots(7,1,figsize=(8,28))
for var,ax in zip(numerical_features,axes.ravel()):
     ax.hist(good_data[var],alpha=0.5)
     ax.hist(bad_data[var],alpha=0.5)
     ax.set_title(var)
     ax.legend(['good_cases','bad_cases'],loc='best')


# In[9]:


display(sns.heatmap(df.isna()))


# In[10]:


df.isna().sum()


# In[11]:


df.checking_status.unique()
df.checking_status.value_counts()
df.checking_status.isna().unique()
sns.displot(df.checking_status)


# In[12]:


#duration
list= np.arange(1000)
plt.figure()
plt.plot(list,df["duration"])
plt.show()


# In[13]:


#plt.figure(figsize=(10,10))
#sns.heatmap(df.isna())


# In[14]:


#credit_history
df.credit_history.value_counts()


# In[15]:


x = df.copy()


# In[16]:


code = {"existing paid":"good","critical/other existing credit":"bad","delayed previously":"bad","all paid":"good","no credits/all paid":"good"}


# In[17]:


x.loc[:,"credit_history"] = x["credit_history"].map(code)


# In[18]:


l= np.sum(x["credit_history"]==x["class"])/1000
print(l)


# In[19]:


#with only this new generated feature we could explain 47% of the target


# In[20]:


#purpose
df["purpose"].unique()


# In[21]:


#savings_status
df["savings_status"].unique()
df["savings_status"].value_counts()


# In[22]:


#'employment'
df["employment"].unique()
df["employment"].value_counts()


# In[23]:


df["class"].value_counts()


# In[24]:


# "employment" alone could explain 65% of all cases
x=df.copy()
code1={"1<=X<4":"good",">=7":"good","4<=X<7":"good","<1":"bad","unemployed":"bad"}
x.loc[:,"employment"] = x["employment"].map(code1)
print(sum(x["employment"]==x["class"])/1000)


# In[25]:


#personal status
len(df["personal_status"].value_counts())


# In[26]:


x=df.copy()
code2={"male single":"good","male mar/wid":"good","male div/sep":"good","female div/dep/mar":"bad"}
x.loc[:,"personal_status"] = x["personal_status"].map(code2)
print(sum(x["personal_status"]==x["class"])/1000)


# In[27]:


df['housing'].value_counts()


# In[28]:


#housing
df['housing'].value_counts()
x=df.copy()
code3={"own":"good","rent":"bad","for free":"good"}
x.loc[:,"housing"] = x["housing"].map(code3)
print(sum(x["housing"]==x["class"])/1000)


# In[29]:


df["class"].value_counts()


# In[30]:


df["job"].value_counts()


# In[31]:


x=df.copy()
code4={"skilled":"good","high qualif/self emp/mgmt":"good","unskilled resident":"bad","unemp/unskilled non res":"bad"}
x.loc[:,"job"] = x["job"].map(code4)
print(sum(x["job"]==x["class"])/1000)


# In[32]:


df['num_dependents'].value_counts()


# In[33]:


df['foreign_worker'].value_counts()

Pair plots; taking insights into the correlations between features.
# In[34]:


sns.pairplot(df[numerical_features])


# In[35]:


sns.displot(df[numerical_features],x='duration',y='credit_amount')
# We notice a positive correlation (0.62)! seems to be logical because generally credits of high amounts are taken for long durations


# In[36]:


np.var(df[numerical_features])


# In[37]:


df[numerical_features].corr()

Only between credit amount & duration we detect an interesting positive correlation
# In[38]:


for column in numerical_features:
    plt.figure(figsize=(3,4))
    sns.violinplot(data=df[numerical_features],x=column,y=df["class"])


# In[39]:


#for column in categorical_features:
#    plt.figure(figsize=(7,3))
#    sns.countplot(data=df[categorical_features],x=column)


# # Preprocessing
Encoding using LabelEncoder. Notice that next we have to scale our data
# In[40]:


from sklearn.preprocessing import LabelEncoder
code={"good":1,"bad":0}
df.loc[:,"class"] = x["class"].map(code)
target = df["class"]
data = df.drop("class",axis=1)
enc= LabelEncoder()
liste = categorical_features.drop("class")
for col in liste: 
    data[col]= enc.fit_transform(data[col])
data

feature selection using tree based model
# In[41]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
x_train,x_test,y_train,y_test=train_test_split(data,target,random_state=0)
tree = DecisionTreeClassifier(random_state=0)
tree.fit(x_train,y_train)
list_impo = tree.feature_importances_
np.median(list_impo,0)
features = data.columns
important_features = [features[i] for i in range(len(features)) if list_impo[i] >= np.median(list_impo,0)]
important_data  = data[important_features]
important_data

Scaling the data using Standard scaler/min_max_scaler
# In[42]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(important_data)
data_preprocessed = data_scaled


# # Splitting the data and Modeling

# In[43]:


x_train,x_test,y_train,y_test=train_test_split(data_preprocessed,target,random_state=0)

Let begin modeling,starting by linear model including linear regressor & ridge regression (with L2 penalisation) & lasso regression (with L1 penalisation which penalises highly the less impoortantes features buut! we don't need that model here because we've already selected the importante features)
# In[44]:


from sklearn.linear_model import LinearRegression ,Ridge
lr = LinearRegression().fit(x_train,y_train)
rr = Ridge(alpha=1).fit(x_train,y_train)

Evaluating the scores of both models
# In[45]:


print(f'the score of the linear regression is.{lr.score(x_test,y_test)}')


# In[46]:


print(f'the score of the Ridge regression is.{rr.score(x_test,y_test)}')

We've got a very low accuracy, let's try another strategy.Let's try now with get_dummies and without scaling
# In[47]:


x_traind,x_testd,y_traind,y_testd=train_test_split(ff,target,random_state=0)
lr = LinearRegression().fit(x_traind,y_traind)
rr = Ridge(alpha=1).fit(x_traind,y_traind)


# In[48]:


print(f'the score of the Ridge regression is.{rr.score(x_testd,y_testd)}')


# In[49]:


print(f'the score of the linear regression is.{lr.score(x_testd,y_testd)}')

sooo sad
Let change the type of the model used, we will try now the tree based models 
for bagging we are going to use the RandomForest classifier
for the boosting we are going to use : Adaboost and gradient boosting
# In[50]:


from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier,GradientBoostingClassifier
rfc = RandomForestClassifier(random_state=0).fit(x_train,y_train)
ada= AdaBoostClassifier(random_state=0).fit(x_train,y_train)
gbc= GradientBoostingClassifier(random_state=0).fit(x_train,y_train)


# In[51]:


print(f'the score of the RandomForestClassifier is.{rfc.score(x_test,y_test)}')
print(f'the score of the adaboost is.{ada.score(x_test,y_test)}')
print(f'the score of the Gradientboostingclassifier is.{gbc.score(x_test,y_test)}')

here we've got a high binary accuracy which is very good for us espicially when using the boosting strategy after a LabelEncoding.
# In[52]:


rfc = RandomForestClassifier(random_state=0).fit(x_traind,y_traind)
ada= AdaBoostClassifier(random_state=0).fit(x_traind,y_traind)
gbc= GradientBoostingClassifier(random_state=0).fit(x_traind,y_traind)
print(f'the score of the RandomForestClassifier is.{rfc.score(x_testd,y_testd)}')
print(f'the score of the adaboost is.{ada.score(x_testd,y_testd)}')
print(f'the score of the Gradientboostingclassifier is.{gbc.score(x_testd,y_testd)}')

after using the one-hot encoding, the accuracy is slightly higher than before and this time the bagging strategy wins. Let try now the MLPs and discuss their performance
# In[53]:


from sklearn.neural_network import MLPClassifier
rn= MLPClassifier(random_state=0,solver='sgd',hidden_layer_sizes=(100,10)).fit(x_traind,y_traind).score(x_testd,y_testd)
rn

the neural_network fails face to tree models 
the model we are keeping tree based models but we still need to find the best parameters
# In[54]:


grid_param = {'n_estimators':np.arange(120,160)}


# In[55]:


from sklearn.model_selection import GridSearchCV
gridsearch = GridSearchCV(RandomForestClassifier(),grid_param).fit(ff,target)
gridsearch


# In[56]:


gridsearch.best_score_


# In[57]:


gridsearch.best_params_


# In[ ]:




