
# coding: utf-8

# In[112]:


import numpy as np
from numpy import nan
import pandas as pd
from pandas import Series, DataFrame
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib

import datetime
from datetime import datetime

import scipy
from scipy import stats

from scipy.stats.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import chi2_contingency

import matplotlib.pyplot as plt
import seaborn as sb
from pylab import rcParams
from pylab import savefig

import statsmodels.api as sm
import statsmodels.formula.api as smf

#importing all packages that I will be using
#**NOTE: DELETE ANYTHING THAT DOESN'T MAKE INTO FINAL VERSION


# In[115]:


#address = '~/Desktop/code/indeedpsci.csv'
address = '//prc-cs-f9dkb42/ecozzolino$/Desktop/code/indeedpsci.csv'
indeed = pd.read_csv(address)
#read in the data from where it is saved on my computer
indeed


# In[114]:


indeed.describe()
#take a look at the columns/features/variables

#My assumptions about the data:
    #1. Date assignment starts: beginning of analysis period
        #seems to be almost uniformly 2/1/2017
    #2. Date assignment ends: end of analysis period
        #varies more, with latest date being 6/19/2017
        #So together, it looks like our time period of analysis is 4 mos in 2017
    #3. Date created: earliest date that advertiser is in the system
        #These are typically years before the period of analysis,
        #though in some cases, they enter the system AFTER the analysis period
        #has begun.
    #4. Assign_days should be the difference between date assingment starts
        #and date assignment ends, if a lead is assigned
            #Alternatively, it could be the difference between the date assignment
            #starts and the first_revenue date, if first_revenue date occurs during
            #the analysis period - effectively truncating the analysis for this 
            #lead, whom we observe until they pay, or until the analysis period ends
        #If unassigned, assign_days should be 0.
    #5. Age should be the difference between date_created and date_assignment begins
        #measuring how long this lead has been around before the period of analysis.
    #6. First revenue date should measure the first time that a lead paid money 
        #to Indeed. In many cases, this first_revenue_date happens before our period
        #of observation begins. This could signify that a lead is good, because they 
        #have paid out in the past, and may have caused them to get assigned at a higher
        #priority than leads without an entry for first revenue date.
        #If first revenue date is missing, I'm assuming it means that this lead has never 
        #brought in revenue to Indeed. 
    #7. Revenue is the amount of money that a lead brought in during the period of analysis.
        #If revenue is missing, I'm assuming that this lead did not bring in any money
        #during the period of analysis.
    


# In[120]:


indeed.date_assignment_starts < indeed.date_assignment_ends
sum(indeed.date_assignment_starts < indeed.date_assignment_ends)
#this is true in all but one case (77,889/77,890)


# In[132]:


pd.to_datetime(indeed.date_assignment_starts)


# In[133]:


pd.to_datetime(indeed.date_assignment_ends)


# In[136]:


pd.to_datetime(indeed.date_created)


# In[137]:


pd.to_datetime(indeed.first_revenue_date)


# In[138]:


#age2 = indeed.date_created - indeed.date_assignment_starts
#Okay, not sure how to test these assumptions. I'll come back to it. 
#https://stackoverflow.com/questions/6749294/understanding-timedelta
#https://docs.python.org/2/library/datetime.html
#https://pandas.pydata.org/pandas-docs/stable/indexing.html


# In[68]:


print(indeed.revenue)


# In[69]:


indeed.duplicated('advertiser_id')
ad_dups = indeed.advertiser_id[indeed.advertiser_id.duplicated()].values
ad_dups
#there are no duplicate entries for advertiser_id
#so, there are 77,890 leads represented in this dataset


# In[70]:


indeed['revenue2']=indeed['revenue'].fillna(0)
#I assume that if the value for 'revenue' is missing, it means
#that there was no revenue during the period of observation. 
#Create a new revenue column, with NaN replaced with 0s
indeed.describe()


# In[71]:


indeed.groupby('assigned').describe()
#40,812 are unassigned
#37,079 are assigned


# In[72]:


indeed.groupby('assigned').mean()
#Comparison of the two groups:
#age of assigned leads is older, number of assign days is smaller, on averge
#revenue is higher among the assigned leads


# In[73]:


rcParams['figure.figsize'] = 5, 4 #this is the size of the plot
sb.set_style('whitegrid') #this is the style: white grid


# In[74]:


indeed.plot(kind='scatter', x='age', y='revenue', c=['darkgray'], s=5)
plt.xlabel('Age of Account')
plt.ylabel('Revenue') 
plt.title('Relationship Between Account Age and Revenue')
plt.show()
#need to fix the sizing - find from DSLynda
#so many 0's that it's hard to see the relationship


# In[75]:


indeed.plot(kind='scatter', x='age', y='revenue2', c=['darkgray'], s=50)
plt.xlabel('Age of Account')
plt.ylabel('Revenue') 
plt.title('Relationship Between Account Age and Revenue2')
plt.show()
#need to fix the sizing - find from DSLynda
#so many 0's that it's hard to see the relationship


# In[76]:


indeed.plot(kind='scatter', x='assign_days', y='revenue', c=['darkgray'], s=50)
plt.xlabel('Number of Days Assigned')
plt.ylabel('Revenue') 
plt.title('Relationship Between Assigned Days and Revenue')
plt.show()
#biggest revenue comes from 0 days assigned
#then spikes in revenue around 60 days, and
#again at the end of the period of observation


# In[77]:


pearsonr_coefficient, p_value = pearsonr(indeed.age, indeed.revenue2)
print('PearsonR Correlation Coefficient %0.3f' % (pearsonr_coefficient))
#significant correlation b/w age and revenue

table = pd.crosstab(indeed.assigned, indeed.revenue)
chi2, p, dof, expected = chi2_contingency(table.values)
print('Chi-square Statistic %0.3f p_value %0.3f' % (chi2, p))
#Revenue of Assigned and unassigned groups significantly different


# In[78]:


indeed['anyfirstrev']=indeed['first_revenue_date'].fillna(0)
indeed['anyfirstrev']
#create indicator for whether an advertising lead has ever given revenue
#replace NaN with 0s - assuming no observations means there hasn't been
#a first revenue date for this adveriser


# In[90]:


indeed['anyfirstrev'] = np.where(indeed['anyfirstrev']!=0, 1, 0)
#complete indicator variable by filling in the dates with 1s
#(indicating yes there has been a first revenue date)


# In[92]:


indeed['anyrev'] = np.where(indeed['revenue2']!=0, 1, 0)
#create an indicator variable for whether or not there's any revenue by a given advertising lead


# In[93]:


indeed


# In[94]:


indeed_trim = indeed[['assigned', 'age', 'assign_days', 'revenue2', 'anyrev', 'anyfirstrev']]
#create dataframe with only the numerical, nonmissing values


# In[95]:


indeed_trim


# In[96]:


indeed_trim.isnull().any()
#making sure we have no missing values in the trim dataset


# In[97]:


X = indeed_trim
sb.pairplot(X)
plt.show()


# In[98]:


corr = X.corr()
corr
sb.heatmap(corr,xticklabels=corr.columns.values, yticklabels=corr.columns.values)
plt.show()
#strongest correlation appears to be between assigned and age


# In[99]:


y = indeed_trim.revenue2
X = indeed_trim.drop(['revenue2', 'anyrev'], axis=1)


# In[100]:


results = sm.OLS(y, X).fit()
results.summary()
#assigned is only marginally significantly predictive of revenue
#meaning: assignment marginally increases revenue
#age does not predict revenue
#assign_days negatively predicts revenue
#Meaning: The longer a lead is assigned, the lower the revenue
#anyfirstrev positively predicts revenue
#meaning: lead having ever given revenue positively predicts revenue within observation period
#R2 is quite low (only 0.03% of variation is explained in this model)


# In[101]:


y = indeed_trim.anyrev
X = indeed_trim.drop(['revenue2', 'anyrev'], axis=1)


# In[104]:


logit = sm.Logit(y, X)
result = logit.fit()
print(result.summary())
#http://blog.yhat.com/posts/logistic-regression-python-rodeo.html


# In[106]:


print(np.exp(result.params))
#R2 is 38% - data are better at explaining binary outcome than amount of revenue
#assignment actually reduces the odds of any revenue by 14%
#any first revenue increases the odds of subsequent revenue astronomically (by 120x)
#age and assign_days both reduce the odds of any revenue, but by small magnitudes (1% and 3% respectively)


# In[108]:


#now, could this be because of some weird values that I was seeing? negative values on age and assign days?
#NOT SURE how to do this.
indeed_trim2 = indeed_trim
indeed_trim2


# In[ ]:


#pd.to_numeric(indeed['anyfirstrev'], errors='coerce')
#indeed['anyfirstrev2']=indeed['anyfirstrev'].fillna(1)
#indeed


# In[ ]:


#if pd.isnull(indeed.first_revenue_date.bool==True):
 #   indeed.anyfirstrev=0
#else:
 #   indeed.anyfirstrev=1
#does not append to indeed


# In[ ]:


#if pd.isnull(indeed.first_revenue_date.bool==True):
 #   indeed['anyfirstrev']=0
#else:
 #   indeed['anyfirstrev']=1
#all rows are 1

