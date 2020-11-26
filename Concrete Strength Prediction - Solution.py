#!/usr/bin/env python
# coding: utf-8

# ## GitHub Project Link - 
# https://github.com/mohansameer1983/concrete-strength-prediction-aiml

# ## Import Libraries

# In[1]:


import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image  
from sklearn import tree
from os import system


# In[2]:


cdf = pd.read_csv("concrete.csv")
cdf.head(10)


# In[3]:


cdf.shape


# In[4]:


cdf.describe()


# In[5]:


cdf.info()  # All columns are numerical, including Target variable


# In[6]:


cdf.nunique()


# ## Univariate Analysis

# In[7]:


# Univariate Analysis
for i in cdf.columns:
    plt.figure(figsize=(15,5))
    sns.scatterplot(data=cdf,y=i,x=cdf.index)
    plt.show()


# # Check Duplicates in data

# In[8]:


##Identify duplicates records in the data
dupes = cdf.duplicated()
sum(dupes)


# In[9]:


##here we can see that there are 25 duplicated rows. We want to remove the duplicate rows.
#Removing Duplicates
cdf =cdf.drop_duplicates()


# In[10]:


##Check duplicates records in the data again
dupes = cdf.duplicated()
sum(dupes)


# In[11]:


## Check missing values
cdf.isnull().values.any()   # Any of the values in the dataframe is a missing value


# In[12]:


cdf.tail()


# # Outliers in Data

# In[13]:


# Boxplot is very helpful in quickly analyzing outliers in data
for i in cdf.columns:
    plt.figure(figsize=(15,5))
    sns.boxplot(x=cdf[i])   # box plot
    plt.show()


# **Note:** Above boxplots shows quite clear outliers in following columns: slag, water, superplastic, fineagg, age
# Let's analyze more and remove significant ones.

# In[14]:


from scipy import stats
import numpy as np
z = np.abs(stats.zscore(cdf))
z


# **Note:** Looking the code and the output above, it is difficult to say which data point is an outlier. Let’s try and define a threshold to identify an outlier.

# In[15]:


threshold = 3
np.where(z > threshold)


# **Note:** The first array contains the list of row numbers and second array respective column numbers, which mean z[21][1] have a Z-score higher than 3.

# In[16]:


print(z[21][1])


# **21st record on column 'slag' is an outlier.**

# In[17]:


Q1 = cdf.quantile(0.25)
Q3 = cdf.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[18]:


np.where((cdf < (Q1 - 1.5 * IQR)) | (cdf > (Q3 + 1.5 * IQR)))


# <span style="font-family: Arial; font-weight:bold;font-size:1.5em;color:#00b3e5;"> Correcting Outliers

# **Note: We are going to try two techniques of optimizing outliers - Z-Score and IQR. After testing with two, IQR method to replace upper outliers with upper whisker gave better model in terms of accuracy. So, you can see we continued with that dataframe from there on.**

# ### Z-Score Removal Technique

# In[27]:


# Z-Score
cdf_1 = cdf[(z < 3).all(axis=1)]    # Select only the rows without a single outlier
cdf_1.shape, cdf.shape


# In[28]:


cdf_z = cdf.copy()   #make a copy of the dataframe

#Replace all the outliers with median values. This will create new some outliers but, we will ignore them

for i, j in zip(np.where(z > threshold)[0], np.where(z > threshold)[1]):# iterate using 2 variables.i for rows and j for columns
    cdf_z.iloc[i,j] = cdf.iloc[:,j].median()  # replace i,jth element with the median of j i.e, corresponding column


# In[29]:


z = np.abs(stats.zscore(cdf_z))
np.where(z > threshold)  # New outliers detected after imputing the original outliers


# ### IQR Removal Technique

# In[30]:


# IQR 
cdf_2 = cdf[~((cdf < (Q1 - 1.5 * IQR)) |(cdf > (Q3 + 1.5 * IQR))).any(axis=1)] # rows without outliers
cdf_2.shape


# In[159]:


cdf_i = cdf.copy()

# Replace every outlier on the lower side by the lower whisker
for i, j in zip(np.where(cdf_i < Q1 - 1.5 * IQR)[0], np.where(cdf_i < Q1 - 1.5 * IQR)[1]): 
    
    whisker  = Q1 - 1.5 * IQR
    cdf_i.iloc[i,j] = whisker[j]
    
    
#Replace every outlier on the upper side by the upper whisker    
for i, j in zip(np.where(cdf_i > Q3 + 1.5 * IQR)[0], np.where(cdf_i > Q3 + 1.5 * IQR)[1]):
    
    whisker  = Q3 + 1.5 * IQR
    cdf_i.iloc[i,j] = whisker[j]
    


# In[160]:


cdf_i.shape, cdf.shape


# In[161]:


cdf_i.info()


# In[34]:


# Boxplots - IQR based
for i in cdf_i.columns:
    plt.figure(figsize=(15,5))
    sns.boxplot(x=cdf_i[i])   # box plot
    plt.show()


# # Bi-Variate Analysis

# In[35]:


# Pairplot is the simplest graphical way to visaully check correlation between different columns of data
sns.pairplot(cdf_i)


# In[162]:


cdf_i.corr()


# In[163]:


sns.heatmap(cdf_i.corr(), annot=True)  # plot the correlation coefficients as a heatmap


# **Note: Above correlation matrix and heatmap of correlation shows following relations between variables:**
# * Negative correlation between superplastic and water
# * Negative correlation between water and fineagg
# * Positive correlation between cement and dependent variable strength. This obviously make sense.
# * Positive correlation between age and dependent variable strength.
# * Positive correlation between superplastic and ash.
# 
# * Coareseagg, fineagg are weak predictors and we can remove them
# * Ash is mild predictor, we can try removing it

# ## Scaling

# In[164]:


cdf_i.columns


# In[165]:


# Scaling helps in normalizing the feature values, which can make otherwise make the model unstable. 
#The default scale for the MinMaxScaler is to rescale variables into the range [0,1]
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame

scaler = MinMaxScaler()

#Keeping separate DF for further processing
cdf_s=cdf_i.copy()

cols_to_scale = ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg',
       'fineagg', 'age', 'strength']

# transform data
cdf_s[cols_to_scale] = scaler.fit_transform(cdf_s[cols_to_scale].to_numpy())

cdf_s.info()


# In[166]:


cdf_s.describe().T


# ## Split Data

# In[167]:


# Get data model ready, where 'strength' column is target variable.
# Dropping feature columns which are not strong predictors
y = cdf_s['strength']
X = cdf_s.drop(['strength','coarseagg','fineagg'], axis=1)
X


# In[168]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=2)


# ## Model Building

# **Let's try following algorithms:**
# 
# * Linear Regression
# * Linear Regression with Polynomial features of degree 2
# * Linear Regression with Polynomial features of degree 3
# * Ridge
# * Lasso
# * Decision Trees
# * Random forest
# * Ada boosting
# * Gradient boosting

# ## Linear Regression

# In[169]:


# Fit a model
from sklearn.linear_model import LinearRegression

regression_model = LinearRegression(fit_intercept=True,normalize=False)
regression_model.fit(X_train, y_train)

print("Linear Regression coefficients - ",regression_model.coef_)


# In[170]:


intercept = regression_model.intercept_

print("The intercept for our model is {}".format(intercept))


# In[171]:


# Publish metrics evaluating model performance

# Train Set
lin_train_score=regression_model.score(X_train, y_train)
print('Linear Reg Score - Train Set -',lin_train_score)

# Test Set
lin_reg=regression_model.score(X_test, y_test)
print('Linear Reg Score - Test Set -',lin_reg)


# ### K-Fold CV - Linear Regression

# In[172]:


# Showing this step separately for Linear Regression. From next model, it will be part of model fit step
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

num_folds = 10
seed = 7

kfold = KFold(n_splits=num_folds, random_state=seed)
results = cross_val_score(regression_model, X, y, cv=kfold, scoring='r2')
print(results)
kf_res_mean=results.mean()*100.0
kf_res_std=results.std()*100.0

print("Accuracy: %.3f%% (%.3f%%)" % (kf_res_mean, kf_res_std))


# In[173]:


#Store the accuracy results for each model in a dataframe for final comparison
resultsDf = pd.DataFrame({'Model':['Linear Regression'],'Training_Score': lin_train_score, 'Test_Score': lin_reg, 
                          'K_Fold_Mean': kf_res_mean, 'K_Fold_Std': kf_res_std})
resultsDf = resultsDf[['Model', 'Training_Score','Test_Score','K_Fold_Mean','K_Fold_Std']]
resultsDf


# ## Linear Regression with Polynomial features of degree 2

# In[174]:


from sklearn.preprocessing import PolynomialFeatures

polynomial_features= PolynomialFeatures(degree=2)

X_poly = polynomial_features.fit_transform(X)

X__poly_train, X_poly_test, y_poly_train, y_poly_test = train_test_split(X_poly, y, test_size=.30, random_state=2)

regression_model_poly_2 = LinearRegression()
regression_model_poly_2.fit(X__poly_train,y_poly_train)

# Train Set
lin_poly_train_score=regression_model_poly_2.score(X__poly_train, y_poly_train)
print('Linear Reg Score - Train Set -',lin_poly_train_score)

# Test Set
lin_reg_poly_test=regression_model_poly_2.score(X_poly_test, y_poly_test)
print('Linear Reg Score - Test Set -',lin_reg_poly_test)

results = cross_val_score(regression_model_poly_2, X_poly, y, cv=kfold, scoring='r2')
print(results)
kf_res_mean=results.mean()*100.0
kf_res_std=results.std()*100.0


# In[175]:


#Store the accuracy results for each model in a dataframe for final comparison
tempResultsDf = pd.DataFrame({'Model':['Linear Regression-Degree 2'],'Training_Score': lin_poly_train_score, 
                              'Test_Score': lin_reg_poly_test, 'K_Fold_Mean': kf_res_mean, 'K_Fold_Std': kf_res_std})
resultsDf = pd.concat([resultsDf, tempResultsDf])
resultsDf = resultsDf[['Model', 'Training_Score','Test_Score','K_Fold_Mean','K_Fold_Std']]
resultsDf


# ## Linear Regression with Polynomial features of degree 3

# In[176]:


polynomial_features= PolynomialFeatures(degree=3)

X_poly = polynomial_features.fit_transform(X)

X__poly_train, X_poly_test, y_poly_train, y_poly_test = train_test_split(X_poly, y, test_size=.30, random_state=2)

regression_model_poly_3 = LinearRegression()
regression_model_poly_3.fit(X__poly_train,y_poly_train)

# Train Set
lin_poly_train_score=regression_model_poly_3.score(X__poly_train, y_poly_train)
print('Linear Reg Score - Train Set -',lin_poly_train_score)

# Test Set
lin_reg_poly_test=regression_model_poly_3.score(X_poly_test, y_poly_test)
print('Linear Reg Score - Test Set -',lin_reg_poly_test)

results = cross_val_score(regression_model_poly_3, X_poly, y, cv=kfold, scoring='r2')
print(results)
kf_res_mean=results.mean()*100.0
kf_res_std=results.std()*100.0


# In[177]:


#Store the accuracy results for each model in a dataframe for final comparison
tempResultsDf = pd.DataFrame({'Model':['Linear Regression-Degree 3'],'Training_Score': lin_poly_train_score, 
                              'Test_Score': lin_reg_poly_test, 'K_Fold_Mean': kf_res_mean, 'K_Fold_Std': kf_res_std})
resultsDf = pd.concat([resultsDf, tempResultsDf])
resultsDf = resultsDf[['Model', 'Training_Score','Test_Score','K_Fold_Mean','K_Fold_Std']]
resultsDf


# **Note:** We can see from above models, that by introducing polynomial feature of high degree, we are getting better accuracy, but that also increasing overfitting, as best fit line is trying to cut through all the data points.

# ## Ridge Regression

# In[178]:


from sklearn.linear_model import Ridge

ridge = Ridge(alpha=.3)
ridge.fit(X_train,y_train)
print ("Ridge Model Coef -", (ridge.coef_))

# Train Set
ridge_reg_train=ridge.score(X_train, y_train)
print('Ridge Regression Score - Train Set -',ridge_reg_train)

# Test Set
ridge_reg_test=ridge.score(X_test, y_test)
print('Ridge Regression Score - Test Set -',ridge_reg_test)

results = cross_val_score(ridge, X, y, cv=kfold, scoring='r2')
print(results)
kf_res_mean=results.mean()*100.0
kf_res_std=results.std()*100.0


# In[179]:


#Store the accuracy results for each model in a dataframe for final comparison
tempResultsDf = pd.DataFrame({'Model':['Ridge Regression'],'Training_Score': ridge_reg_train, 
                              'Test_Score': ridge_reg_test, 'K_Fold_Mean': kf_res_mean, 'K_Fold_Std': kf_res_std})
resultsDf = pd.concat([resultsDf, tempResultsDf])
resultsDf = resultsDf[['Model', 'Training_Score','Test_Score','K_Fold_Mean','K_Fold_Std']]
resultsDf


# ## Lasso Regression

# In[181]:


from sklearn.linear_model import Lasso

cdf_l = cdf_i.copy()
y1 = cdf_l['strength']
X1 = cdf_l.drop(['strength','coarseagg','fineagg'], axis=1)
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=.30, random_state=2)

lasso = Lasso(alpha=0.1,normalize=False)
lasso.fit(X1_train,y1_train)
print ("Lasso Model Coef -", (lasso.coef_))

# Train Set
lasso_reg_train=lasso.score(X1_train, y1_train)
print('Lasso Regression Score - Train Set -',lasso_reg_train)

# Test Set
lasso_reg_test=lasso.score(X1_test, y1_test)
print('Lasso Regression Score - Test Set -',lasso_reg_test)

results = cross_val_score(lasso, X1, y1, cv=kfold, scoring='r2')
print(results)
kf_res_mean=results.mean()*100.0
kf_res_std=results.std()*100.0


# In[182]:


#Store the accuracy results for each model in a dataframe for final comparison
tempResultsDf = pd.DataFrame({'Model':['Lasso Regression'],'Training_Score': lasso_reg_train, 
                              'Test_Score': lasso_reg_test, 'K_Fold_Mean': kf_res_mean, 'K_Fold_Std': kf_res_std})
resultsDf = pd.concat([resultsDf, tempResultsDf])
resultsDf = resultsDf[['Model', 'Training_Score','Test_Score','K_Fold_Mean','K_Fold_Std']]
resultsDf


# <a id  = ensemblelearning></a>
# #                             Ensemble Technique - Bagging

# In[197]:


from sklearn.ensemble import BaggingRegressor

bgcl = BaggingRegressor(n_estimators=10,random_state=1)

bgcl = bgcl.fit(X_train, y_train)


# In[198]:



y_predict = bgcl.predict(X_test)
acc_BG_train=bgcl.score(X_train , y_train)
print("Bagging - Train Accuracy:",acc_BG_train)
acc_BG = bgcl.score(X_test , y_test)
print("Bagging - Test Accuracy:",acc_BG)

results = cross_val_score(bgcl, X, y, cv=kfold, scoring='r2')
print(results)
kf_res_mean=results.mean()*100.0
kf_res_std=results.std()*100.0


# In[200]:


#Store the accuracy results for each model in a dataframe for final comparison
tempResultsDf = pd.DataFrame({'Model':['Bagging'],'Training_Score': acc_BG_train, 
                              'Test_Score': acc_BG, 'K_Fold_Mean': kf_res_mean, 'K_Fold_Std': kf_res_std})
resultsDf = pd.concat([resultsDf, tempResultsDf])
resultsDf = resultsDf[['Model', 'Training_Score','Test_Score','K_Fold_Mean','K_Fold_Std']]
resultsDf


# # Ensemble Technique - AdaBoosting

# In[201]:


from sklearn.ensemble import AdaBoostRegressor
abcl = AdaBoostRegressor(n_estimators=50,random_state=1)
abcl = abcl.fit(X_train, y_train)


# In[204]:


y_predict = abcl.predict(X_test)
abcl_train=abcl.score(X_train , y_train)
print("Ada Boosting - Train Accuracy:",abcl_train)
abcl_test = abcl.score(X_test , y_test)
print("Ada Boosting - Test Accuracy:",abcl_test)

results = cross_val_score(abcl, X, y, cv=kfold, scoring='r2')
print(results)
kf_res_mean=results.mean()*100.0
kf_res_std=results.std()*100.0


# In[203]:


#Store the accuracy results for each model in a dataframe for final comparison
tempResultsDf = pd.DataFrame({'Model':['Ada Boosting'],'Training_Score': abcl_train, 
                              'Test_Score': abcl_test, 'K_Fold_Mean': kf_res_mean, 'K_Fold_Std': kf_res_std})
resultsDf = pd.concat([resultsDf, tempResultsDf])
resultsDf = resultsDf[['Model', 'Training_Score','Test_Score','K_Fold_Mean','K_Fold_Std']]
resultsDf


# #                     Ensemble Technique - GradientBoost

# In[206]:


from sklearn.ensemble import GradientBoostingRegressor
gbcl = GradientBoostingRegressor(n_estimators = 50,random_state=1)
gbcl = gbcl.fit(X_train, y_train)


# In[207]:


y_predict = gbcl.predict(X_test)
gbcl_train=gbcl.score(X_train , y_train)
print("Gradient Boost - Train Accuracy:",gbcl_train)
gbcl_test = gbcl.score(X_test , y_test)
print("Gradient Boost - Test Accuracy:",gbcl_test)

results = cross_val_score(gbcl, X, y, cv=kfold, scoring='r2')
print(results)
kf_res_mean=results.mean()*100.0
kf_res_std=results.std()*100.0


# In[208]:


#Store the accuracy results for each model in a dataframe for final comparison
tempResultsDf = pd.DataFrame({'Model':['Gradient Boosting'],'Training_Score': gbcl_train, 
                              'Test_Score': gbcl_test, 'K_Fold_Mean': kf_res_mean, 'K_Fold_Std': kf_res_std})
resultsDf = pd.concat([resultsDf, tempResultsDf])
resultsDf = resultsDf[['Model', 'Training_Score','Test_Score','K_Fold_Mean','K_Fold_Std']]
resultsDf


# # Ensemble RandomForest Classifier

# In[209]:


from sklearn.ensemble import RandomForestRegressor
rfcl = RandomForestRegressor(n_estimators = 50, random_state=1)
rfcl = rfcl.fit(X_train, y_train)


# In[210]:


y_predict = rfcl.predict(X_test)
rfcl_train=bgcl.score(X_train , y_train)
print("Random Forest - Train Accuracy:",rfcl_train)
rfcl_test = bgcl.score(X_test , y_test)
print("Random Forest - Test Accuracy:",rfcl_test)

results = cross_val_score(rfcl, X, y, cv=kfold, scoring='r2')
print(results)
kf_res_mean=results.mean()*100.0
kf_res_std=results.std()*100.0


# In[211]:


#Store the accuracy results for each model in a dataframe for final comparison
tempResultsDf = pd.DataFrame({'Model':['Random Forest'],'Training_Score': rfcl_train, 
                              'Test_Score': rfcl_test, 'K_Fold_Mean': kf_res_mean, 'K_Fold_Std': kf_res_std})
resultsDf = pd.concat([resultsDf, tempResultsDf])
resultsDf = resultsDf[['Model', 'Training_Score','Test_Score','K_Fold_Mean','K_Fold_Std']]
resultsDf


# ## Hyperparameter Tuning
# 
# We will use GridsearchCV and RandomSearchCV for tuning hyperparameters

# ## GridSearch CV - RandomForest Model

# In[212]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict,StratifiedKFold
params = {
    'bootstrap': [True,False],
    'min_samples_leaf': [1,3],
    'min_samples_split': [2,3],
    'n_estimators': [25,50,100]
}    
kfold = KFold(n_splits=10, random_state=7)
grid = GridSearchCV(estimator = rfcl, param_grid = params, cv=kfold)

grid.fit(X_train,y_train)


# In[213]:


grid_train=grid.score(X_train , y_train)
print("GridSearchCV - Train Accuracy:",grid_train)
grid_test = grid.score(X_test , y_test)
print("GridSearchCV - Test Accuracy:",grid_test)


# In[214]:


#Store the accuracy results for each model in a dataframe for final comparison
tempResultsDf = pd.DataFrame({'Model':['GridSearchCV-Random Forest'],'Training_Score': grid_train, 
                              'Test_Score': grid_test, 'K_Fold_Mean':'NA', 'K_Fold_Std': 'NA'})
resultsDf = pd.concat([resultsDf, tempResultsDf])
resultsDf = resultsDf[['Model', 'Training_Score','Test_Score','K_Fold_Mean','K_Fold_Std']]
resultsDf


# ## RandomSearchCV - Ada Boosting Model

# In[217]:


from sklearn.model_selection import RandomizedSearchCV
param_dist = {
 'n_estimators': [50, 100],
 'learning_rate' : [0.01,0.05,0.1,0.3,1],
 'loss' : ['linear', 'square', 'exponential']
 }   
grid1 = RandomizedSearchCV(estimator = abcl,param_distributions = param_dist,
 cv=10,
 n_iter = 10,
 n_jobs=-1)

grid1.fit(X_train,y_train)


# In[218]:


grid1.best_params_


# In[219]:


grid1_train=grid1.score(X_train , y_train)
print("RandomSeachCV - Train Accuracy:",grid1_train)
grid1_test = grid1.score(X_test , y_test)
print("RandomSearchCV - Test Accuracy:",grid1_test)


# In[220]:


#Store the accuracy results for each model in a dataframe for final comparison
tempResultsDf = pd.DataFrame({'Model':['RandomSearchCV-Ada Boosting'],'Training_Score': grid1_train, 
                              'Test_Score': grid1_test, 'K_Fold_Mean': 'NA', 'K_Fold_Std': 'NA'})
resultsDf = pd.concat([resultsDf, tempResultsDf])
resultsDf = resultsDf[['Model', 'Training_Score','Test_Score','K_Fold_Mean','K_Fold_Std']]
resultsDf


# ## Conclusion
# * From above list looks like GridSearchCV using Random Forest Model have the best result.
# * Target variable 'strength' is most  strongly predicted using cement, water, superplastic, age and slag.
