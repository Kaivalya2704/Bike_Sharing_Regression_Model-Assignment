#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import statsmodels.api as sm
import statsmodels
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore') 


# In[2]:


# Data understanding and data loading


# In[3]:


#reading the file
bikes=pd.read_csv('day.csv')
bikes.head(10)


# In[4]:


#shape of the data
bikes.shape


# In[5]:


#info of the data
bikes.info()


# In[6]:


#info of the data
bikes.info()


# In[7]:


#using describe to summarize the data
bikes.describe()


# **FROM THE ABOVE, WE GET THE FOLLOWING INSIGHTS :**
# 
#     We can drop three columns casual,registered and index (given they are not features)
#     
#     There are no null values as you can see from the count
#     
#     And the target variable is the count, 'cnt' in the dataframe

# In[9]:


#looking at the columns
bikes.columns


# In[10]:


count=bikes.isnull().sum()
print(count)


# In[11]:


#Lets drop the columns we dont need, casual registered and index. As casual + registered = cnt, we dont need them and instant as index
bikes.drop(['instant'],axis=1,inplace=True)
bikes.head()


# In[12]:


#drop dteday as it is same as yr and mnth
bikes.drop(['dteday'],axis=1,inplace=True)
bikes.head()


# In[13]:


#Casual+ Registered is cnt, thus we can drop them
bikes.drop(['casual','registered'],axis=1,inplace=True)
bikes.head()


# In[14]:


#Now from info, we have a few categorical columns. Lets change them to binary values so that they are a better fit for the model and visualization as well


# In[15]:


#replacing season
bikes['season'].replace({1:"spring",2:"summer",3:"fall",4:"winter"},inplace=True)
bikes.head(5)


# In[16]:


#replacing weathersit
bikes['weathersit'].replace({1:"Clear_Few Clouds",2:"Mist_cloudy",3:"Light rain_Light snow_Thunderstorm",4:'Heavy Rain_Ice Pallets_Thunderstorm_Mist'},inplace=True)
bikes.head(5)


# In[17]:


#replacing weekday
bikes['weekday'].replace({0:"Sunday",1:"Monday",2:"Tuesday",3:"Wednesday",4:"Thursday",5:"Friday",6:"Saturday"},inplace=True)
bikes.head(5)


# In[18]:


#changing dataypes of numerical columns
bikes[['temp','atemp','hum','windspeed','cnt']]=bikes[['temp','atemp','hum','windspeed','cnt']].apply(pd.to_numeric)
bikes.head()


# In[19]:


bikes.info()


# In[20]:


# EDA for Categorical and Numerical Variables


# For categorical values:
# 
# Visualising count using seasons,weather,year etc

# In[21]:


#Visualising categorical Variables to understand data better
sns.set_palette("hls", 8)
plt.figure(figsize=(30, 15))
plt.subplot(3,3,1)
sns.boxplot(x = 'season', y = 'cnt', data = bikes)#yr','mnth','workingday','weathersit','weekday'
plt.subplot(3,3,2)
sns.boxplot(x = 'yr', y = 'cnt', data = bikes)
plt.subplot(3,3,3)
sns.boxplot(x = 'mnth', y = 'cnt', data = bikes)
plt.subplot(3,3,4)
sns.boxplot(x = 'workingday', y = 'cnt', data = bikes)
plt.subplot(3,3,5)
sns.boxplot(x = 'weathersit', y = 'cnt', data = bikes)
plt.subplot(3,3,6)
sns.boxplot(x = 'weekday', y = 'cnt', data = bikes)
plt.subplot(3,3,7)
sns.boxplot(x = 'holiday', y = 'cnt', data = bikes)
plt.show()


# **Some insights from the above visualizations are :** 
#     
#     In fall, there seems to be highest demand of rented the bikes, followed by Summer and Winter
#     
#      Spring seems to be the least season where people rent bikes     
#      
#      December, January, February have the least demand probably due to winter season
#      
#      There are similar demands whether it's a working day or not.
#      
#      Forecasting weather plays a big role in the bikes demand, more clearer the weather, more the demand

# In[22]:


#For Numerical Values


# In[23]:


sns.pairplot(bikes, vars=['temp','atemp','hum','windspeed',"cnt"])
plt.show()


# In[24]:


#From the above plots, its clear that atemp and temp have a relation
#Thus linear regression can be performed on the same

#Correlation using heatmaps
plt.figure(figsize = (16, 10))
sns.heatmap(bikes.corr(), annot = True, cmap="Reds")
plt.show()


# In[25]:


#The correlation between temp and temp is almost 1, so we can drop one of them and focus on another column. Lets drop temp and use atemp

bikes.drop(['temp'],axis=1,inplace=True)
bikes.head(10)


# In[26]:


# Pre Processing/Data Preperation


# In[27]:


#Lets Create dummy variables for all categorical variables having no. of column > 2, using Use pd.get_dummies([df.season], drop_first=True]


# In[28]:


#Convert variables to object type
bikes['mnth']=bikes['mnth'].astype(object)
bikes['season']=bikes['season'].astype(object)
bikes['weathersit']=bikes['weathersit'].astype(object)
bikes['weekday']=bikes['weekday'].astype(object)
bikes.info()


# In[29]:


#CREATING DUMMY VARIABLES FOR CATEGORICAL DATA 

Season_condition=pd.get_dummies(bikes['season'],drop_first=True)
Weather_condition=pd.get_dummies(bikes['weathersit'],drop_first=True)
Day_of_week=pd.get_dummies(bikes['weekday'],drop_first=True)
Month=pd.get_dummies(bikes['mnth'],drop_first=True)


# In[30]:


#Lets now concat the made dummies to the dataset
bikes=pd.concat([bikes,Season_condition],axis=1)
bikes=pd.concat([bikes,Day_of_week],axis=1)
bikes=pd.concat([bikes,Weather_condition],axis=1)
bikes=pd.concat([bikes,Month],axis=1)
bikes.info()


# In[31]:


#Lets now delete the original columns for the replaced dummy columns
bikes.drop(['season'],axis=1,inplace=True)
bikes.drop(['weathersit'],axis=1,inplace=True)
bikes.drop(['weekday'],axis=1,inplace=True)
bikes.drop(['mnth'],axis=1,inplace=True)
bikes.head()


# In[32]:


# TRAIN AND TEST SPLIT


# Now, lets split the data into train and test, with a random split. 
# 
# The normal standards can vary, and lets go for 80% and 20%  split

# In[33]:


#We use sklearn and statsmodel, which we have already imported at the start
#For the random variable
np.random.seed(0)
bikes_train, bikes_test = train_test_split(bikes, train_size=0.80, random_state=100)
bikes_train.head()


# In[34]:


#lets check the columns with the split
print(bikes_train.shape)
print(bikes_test.shape)


# In[35]:


bikes_train.head()


# In[36]:


bikes_train.columns


# In[37]:


#From above we can see that the split has happened properly.Lets now Scale the data for all the numerical variables wihout count
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()


# In[39]:


num_vars=['atemp','hum','windspeed','cnt']
bikes_train[num_vars] = scaler.fit_transform(bikes_train[num_vars])
bikes_train.describe()


# In[40]:


#Lets check the multicolinearity between the variables
plt.figure(figsize = [25,20])
sns.heatmap(bikes_train.corr(),annot =True, cmap= 'Reds')
plt.show()


# In[41]:


# MODEL BUILDING


# In[42]:


#Now with the above done and ready, we can start with the model building. Lets now divide the training sets into X_train and y_train sets
y_train = bikes_train.pop('cnt')
X_train = bikes_train


# In[43]:


print(X_train.shape)
print(y_train.shape)


# In[44]:


X_train.head()


# In[45]:


y_train.head()


# In[46]:


#Feature Selection using RFE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

#USING RFE APPROACH FOR FEATURE SELECTION
# WE START WITH 15 VARS AND WILL USE MIXED APPROACH TO BUILD A MODEL

lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm,n_features_to_select=15)            
rfe = rfe.fit(X_train, y_train)


# In[47]:


list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# In[48]:


col = X_train.columns[rfe.support_]
col


# In[49]:


X_train.columns[~rfe.support_]


# In[50]:


X_train_rfe = X_train[col]


# In[51]:


#BUILDING MODEL USING STATSMODEL:

import statsmodels.api as sm  
X_train_rfe1 = sm.add_constant(X_train_rfe)

lm = sm.OLS(y_train,X_train_rfe1).fit()

print(lm.summary())


# In[52]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[53]:


X_train_rfe1.head()


# In[54]:


#COLUMN hum HAS A VERY HIGH VIF SO WE DROP IT 
X_train_rfe=X_train_rfe.drop(['hum'],axis=1)

import statsmodels.api as sm  
X_train_rfe1 = sm.add_constant(X_train_rfe)

lm1 = sm.OLS(y_train,X_train_rfe1).fit()

print(lm1.summary())    


# In[55]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[56]:


#temp column has a high vif, thus we drop it
X_train_rfe=X_train_rfe.drop(['atemp'],axis=1)

 
X_train_rfe2 = sm.add_constant(X_train_rfe)
lm2 = sm.OLS(y_train,X_train_rfe2).fit()
print(lm2.summary())


# In[57]:


vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[58]:


X_train_rfe.columns


# In[59]:


#Windspeed has high VIF >2 so we drop it
X_train_rfe=X_train_rfe.drop(['windspeed'],axis=1)

X_train_rfe3 = sm.add_constant(X_train_rfe)
lm3 = sm.OLS(y_train,X_train_rfe3).fit()
print(lm3.summary())


# In[60]:


#Winter has A VERY HIGH p-value WHUCH MEANS IT IS insignificant SO WE DROP IT
X_train_rfe=X_train_rfe.drop(['winter'],axis=1)

X_train_rfe4 = sm.add_constant(X_train_rfe)
lm4 = sm.OLS(y_train,X_train_rfe4).fit()
print(lm4.summary())


# In[61]:


#ADDING SATURDAY AND CHECKING IF MODEL IMPROVES

X_train_rfe['Saturday']=X_train['Saturday']
X_train_rfe.head()
X_train_rfe5 = sm.add_constant(X_train_rfe)
lm5 = sm.OLS(y_train,X_train_rfe5).fit()
print(lm5.summary())


# In[62]:


#We drop the column with a high p-value, thus saturday
X_train_rfe=X_train_rfe.drop(['Saturday'],axis=1)


# In[63]:


X_train_rfe6= sm.add_constant(X_train_rfe)
lm6 = sm.OLS(y_train,X_train_rfe6).fit()
print(lm6.summary())


# In[64]:


#Lets add sunday to check if the model improves
X_train_rfe['Sunday']=X_train['Sunday']
X_train_rfe.head()


# In[65]:


X_train_rfe7 = sm.add_constant(X_train_rfe)
lm7 = sm.OLS(y_train,X_train_rfe7).fit()
print(lm7.summary())


# In[66]:


X_train_rfe7 = sm.add_constant(X_train_rfe)
lm7 = sm.OLS(y_train,X_train_rfe7).fit()
print(lm7.summary())


# In[67]:


#lets add working day to check if there is an improvement in the model
X_train_rfe['workingday']=X_train['workingday']
X_train_rfe.head()


# In[68]:


X_train_rfe8 = sm.add_constant(X_train_rfe)
lm8 = sm.OLS(y_train,X_train_rfe8).fit()
print(lm8.summary())


# In[69]:


vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[70]:


#Lets add month 7 and see if it helps with the model
X_train_rfe[7]=X_train[7]
X_train_rfe.head()


# In[71]:


X_train_rfe9 = sm.add_constant(X_train_rfe)
lm9 = sm.OLS(y_train,X_train_rfe9).fit()
print(lm9.summary())


# In[72]:


vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[73]:


#As we can see the model has improved, we add month 10 to check if it helps more
X_train_rfe[10]=X_train[10]
X_train_rfe.head() 


# In[74]:


X_train_rfe10 = sm.add_constant(X_train_rfe)
lm10 = sm.OLS(y_train,X_train_rfe10).fit()
print(lm10.summary())


# In[75]:


vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[76]:


#As model shows month 10 shows it helps, lets try to add the 11th month
X_train_rfe[11]=X_train[11]
X_train_rfe.head()


# In[77]:


X_train_rfe12 = sm.add_constant(X_train_rfe)
lm12 = sm.OLS(y_train,X_train_rfe12).fit()
print(lm12.summary())


# In[78]:


#Due to the above, we drop the 11th month


# In[79]:


#Lets see if we can add 12th month to improve the model
X_train_rfe[12]=X_train[12]
X_train_rfe.head()


# In[80]:


X_train_rfe13 = sm.add_constant(X_train_rfe)
lm13 = sm.OLS(y_train,X_train_rfe13).fit()
print(lm13.summary())


# In[81]:


#We see high p-value for 12 so we drop it

X_train_rfe=X_train_rfe.drop([12],axis=1)


# Month 10th is where the pvalue is not high and we can see the model improving, thus we can take lm10 as model with the best results

# **PREDICTING THE VALUES AND CALCULATING RESIDUALS**
# 

# In[82]:


#predicting the values 
y_train_cnt = lm10.predict(X_train_rfe10)


# In[83]:


#calculating the residuals
res = y_train - y_train_cnt


# In[84]:


# Plot the histogram of the error terms
fig = plt.figure(figsize=[7,5])
sns.distplot((res), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                   
plt.xlabel('Errors', fontsize = 18)                         
plt.show


# In[85]:


X_train_rfe10.columns


# In[86]:


print(X_train_rfe10.shape)
print(res.shape)


# In[88]:


#Scaling the test data

num_vars=['atemp','hum','windspeed','cnt']
bikes_test[num_vars] = scaler.fit_transform(bikes_test[num_vars])


# In[90]:


#Creating x and y sets

y_test = bikes_test.pop('cnt')
X_test = bikes_test


# In[92]:


print(y_test.shape)
print(X_test.shape)


# In[95]:


#Selecting the variables that were part of final model (Model 8).
X_train_new=X_train_rfe10.drop(['const'], axis=1)


# In[97]:


# Now let's use our model to make predictions.
# Creating X_test_new dataframe by dropping variables from X_test

X_test_new = X_test[X_train_new.columns]

# Adding a constant variable 
X_test_new = sm.add_constant(X_test_new)

X_train_rfe10.columns


# In[110]:


#predicting on the basis of test data
y_pred = lm10.predict(X_test_new)   


# **Check for Homoscedasticity**

# In[111]:


plt.figure(figsize = [8,5])
p = sns.scatterplot(y_train_cnt,res)
plt.xlabel('y_pred/predicted values')
plt.ylabel('Residuals')

p = sns.lineplot([0,1],[0,0],color='red')
p = plt.title('Residuals vs fitted values', fontsize = 20)


# The plot above shows that the residuals have almost equal variances from the regression lines

# **EVALUATING THE MODEL**

# In[112]:


# Plotting y_test and y_test_pred to understand the spread

fig = plt.figure()
plt.scatter(y_test, y_pred, alpha=.5)
fig.suptitle('y_test vs y_test_pred', fontsize = 20)              
plt.xlabel('y_test', fontsize = 18)                     
plt.ylabel('y_pred', fontsize = 16) 


# A linear relation can be seen between the y_test and y_pred. Thus the model can explain the change in demand properly

# **RESIDUAL ANALYSIS**

# In[113]:


from sklearn.metrics import r2_score
r2_test = r2_score(y_test, y_pred)
r2_train = r2_score(y_train, y_train_cnt)

print('Test data r^2 :',round((r2_test*100),2))
print('Train data r^2 :',round((r2_train*100),2))


# **Adjusted R^2 Value for TEST**

# In[115]:


# n for test data ,n1 for train data is number of rows
n = X_test.shape[0]
n1 = X_train_rfe10.shape[0]

# Number of features (predictors, p for test data, p1 for train data) is the number of columns
p = X_test.shape[1]
p1 = X_train_rfe10.shape[1]


# We find the Adjusted R-squared using the formula

adjusted_r2_test = 1-(1-r2_test)*(n-1)/(n-p-1)
adjusted_r2_train = 1-(1-r2_train)*(n1-1)/(n1-p1-1)

print('Test data adjusted r^2 :',round((adjusted_r2_test*100),2))
print('Train data adjusted r^2 :',round((adjusted_r2_train*100),2))


# **Lets Compare the R^2 and Adjustd R^2 Values for the Model**

# * Test data r^2 : 76.89
# 
# * Train data r^2 : 76.83
# 
# * Test data adjusted r^2 : 71.36
# 
# * Train data adjusted r^2 : 76.22
# 

# From the r2 values and adjusted r2 values, we can say that the model is good enough for the current days dataset

# **FROM THE ABOVE MODEL, THE VARIABLES THAT DECIDE THE DEMANDS OF BIKES ARE :**
# * Month of March, June and September (Increase in Demand)*
# * Spring Season (Decrease in Demand)*
# * If its a holiday, especially  Sunday (Decrease in Demand)*
# * If the weather is clear/Good (Increase in Demand)*
# * Temperature (Increase/Decrease)*
# * Year 2019 *

# In[ ]:




