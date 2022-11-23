#!/usr/bin/env python
# coding: utf-8

# # Hospital costs
# &ensp; Name:        hospital_case.ipynb <br>
# &ensp; Description: To calculate a hospital's allowed cost, make a model for predicting the average cost per patient <br>
# 

# # Import Packages

# In[1]:


get_ipython().system('pip install xgboost')

import numpy as np 
import pandas as pd 
import os 
from sklearn.model_selection import train_test_split #single train/test split
import statsmodels.api as sm # statsmodel over sklearn for model summary function
import matplotlib.pyplot as plt 
import seaborn as sns 
import xgboost as xgb
import warnings
warnings.simplefilter("ignore", UserWarning)


# In[2]:


os.chdir(r'C:\Users\M-P\Downloads')


# ## Load Data

# In[3]:


# Read data file
data = pd.read_excel("C:\Users\M-P\Downloads\210920-Hospital case Data.xlsx" )

data.head()


# ## Inspection and descriptives of data
# &ensp; - Finding relevant variables, sorting categorical values <br>
# 

# In[4]:


#variables in the data frame
list(data.columns)


# In[5]:


# Summarize the data frame:
display(data.describe())

# Summary of non-categorical variables
list_cat_vars = [col for col in data.select_dtypes(include=['object']).columns]
print("The following columns are catagorical variables:",list_cat_vars)


# In[7]:


# Plot the number of observations per category
def plotCategoryCount(data, var, width = 15, heigth = 5):
    fig, ax = plt.subplots()
    fig.set_size_inches(width, heigth) 
    category_counts = data[var].value_counts()
    sns.set(style="darkgrid")
    sns.barplot(category_counts.index, category_counts.values, alpha=0.9)
    plt.title('Frequency Distribution of ' + var)
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel(var, fontsize=12)
    plt.show()


# In[8]:


#Distribution of observations over categories
display(plotCategoryCount(data, 'HOSPITAL', width = 16))


# In[9]:


# Check for columns with missing values
list_col_with_missing = [col for col in data.columns if data[col].isnull().any()]
print("The following columns contain missing values:",list_col_with_missing)

# Get histograms of continuous data
_ = plt.hist(data["AGE"])
plt.title('Histogram of age per patient')
plt.show()

_ = plt.hist(data["COST"], bins='auto')
plt.title('Histogram of cost per patient')
plt.show()


# ## Split data into target and predictor variables
# 

# In[10]:


# Split data into X (all explanatory variables) and y (variable to be predicted by X, i.e. the cost)
X = data.drop(["COST"], axis = 1) 
y = data["COST"]

print(data.columns)

# Variables to include in model, split into continuous and categorical
continuous_vars_to_incl = [] # Age is already included as categorical variable
categorical_vars_to_incl = ['GENDER', 'AGE_GROUP', 'SOCIAL_CLASS', 'DIAGNOSIS_GROUP']

# Dummify all categorical variables
dummies = pd.get_dummies(X[categorical_vars_to_incl], drop_first = True)
X = pd.concat([X[continuous_vars_to_incl], dummies], axis=1)


# ## Fit Model
# * Fit linear model (statistics) <br>
# * Fit XGBoost Model (machine learning)  <br>

# ### Linear Model

# In[11]:


# For Linear Regression (OLS) we first need to add a constant
X_OLS = sm.add_constant(X)

# Fit a model
linear_model = sm.OLS(endog=y, exog=X_OLS).fit()

# Calculate fitted values
y_pred_linear = linear_model.predict(X_OLS) 

# Show the model's coefficients and other information
linear_model.summary()


# In[12]:


# Add linear model predictions to data
data['OLS_pred'] = y_pred_linear


# In[39]:


# Calculate average cost and prediction per hospital, and display results
results = data.groupby('HOSPITAL').agg(
    n =pd.NamedAgg(column="COST", aggfunc="count"),
    mean_COST = pd.NamedAgg(column="COST", aggfunc="mean"),
    mean_OLS_pred =pd.NamedAgg(column="OLS_pred", aggfunc="mean")
    )
results.round(2)


# ### XGBoost Model

# We can try to get an even better fit by using a non-linear model like XGBoost.

# In[40]:


# Initialize XGBoost model
xgb_model= xgb.XGBRegressor(max_depth=10, min_child_weight=7, objective='reg:squarederror', verbose=False, feval='xgb_mape',
                             n_estimators=50)

# Train the model using the training sets
# For linear model we need to drop one of the 3 class dummies to overcome the matrix
xgb_model.fit(X,y)

# Make predictions using the testing set
y_pred_xgb = xgb_model.predict(X)


# In[15]:


# Add linear model predictions to data
data['XGB_pred'] = y_pred_xgb


# In[16]:


# Calculate average per hospital and display results
results = data.groupby('HOSPITAL').agg(
    n =pd.NamedAgg(column="COST", aggfunc="count"),
    mean_COST = pd.NamedAgg(column="COST", aggfunc="mean"),
    mean_OLS_pred =pd.NamedAgg(column="OLS_pred", aggfunc="mean"),
    mean_XGB_pred =pd.NamedAgg(column="XGB_pred", aggfunc="mean"),
)
results.round(2)


# In[ ]:





# In[ ]:




