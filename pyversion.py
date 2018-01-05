
# coding: utf-8

# In[1]:


#To begin, I'll import the packages that I'll be using.

import numpy as np
from numpy import nan
import pandas as pd

import scipy
from scipy import stats
from scipy.stats.stats import pearsonr
from scipy.stats import chi2_contingency

import matplotlib.pyplot as plt
import seaborn as sb

import statsmodels.api as sm
import statsmodels.formula.api as smf


# In[2]:


#Next, I'll read in the data from where it's saved on my computer.
address = '//prc-cs-f9dkb42/ecozzolino$/Desktop/code/indeedpsci.csv'
indeed = pd.read_csv(address)
indeed


# In[3]:


#My understanding of the situation:
# Indeed has a system that routes leads to the sales team based on an
# estimate of the potential quality of each lead. Leads are assigned from highest to 
# lowest probability until each sales rep has a full sales book. 

#I don't have access to this probability. But, whether or not a lead
# is assigned - and the amount of time that a lead is assigned - each might reflect
# this underlying latent probability. Leads that are assigned should be assigned 
# because of their higher underlying probability of yielding revenue. Of the assigned
# leads, assign_days could reflect this probability too, with some leads
# assigned later in the process because they were a lower priority.

#If the assigned leads yield revenue and the unassigned leads do not,
# this is an indication that the prioritization has worked well. But, assigned
# leads that do not yield revenue and unassigned leads that do yield revenue each
# speak to a potential weakness in this prioritization scheme.

#Now, I'll take a look at the variables in this dataset.
indeed.describe()

#My assumptions about the data:
    #1. Date assignment starts = beginning of analysis period
        #seems to be almost uniformly 2/1/2017
    #2. Date assignment ends = end of analysis period
        #This varies more, with latest date looking like 6/19/2017
        #So together, it looks like our time period of analysis is 4 months in 2017.
    #3. Date created = earliest date that advertiser is in the system
        #These are typically years before the period of analysis,
        #though in some cases, they enter the system AFTER the analysis period
        #has begun.
    #4. Assign_days = the difference between date assingment starts
        #and date assignment ends, if a lead is assigned.
            #Alternatively, it could be the difference between the date assignment
            #starts and the first_revenue_date, if first_revenue_date occurrs during
            #the analysis period - effectively truncating the analysis for this 
            #lead, who we observe until they pay, or until the analysis period ends.
            #If a lead is unassigned, it's value on assign_days should be 0.
    #5. Age should be the difference between date_created and date_assignment_starts
        #measuring how long this lead has been around before the period of analysis.
    #6. First revenue date should measure the first time that a lead paid money 
        #to Indeed. In many cases, this first_revenue_date happens before our period
        #of observation begins. I would guess that these leads are higher priority, 
        #because they have paid out in the past. I would expect more of the leads with
        #a non-missing first_revenue_date to be assigned.
          #If first revenue date is missing, I'm assuming it means that this lead has never 
          #previously brought in revenue to Indeed. 
    #7. Revenue is the amount of money that a lead brought in during the period of analysis.
        #If revenue is missing, I'm assuming that this lead did not bring in any money
        #during the period of analysis.
  


# In[4]:


#Before answering the questions, I want to take care of the missing data.
indeed.isnull().any()
# The variables 'first_revenue_date' and 'revenue' both have missing values.
# I'm assuming that these values should be 0, because there has not been
# any revenue - either prior to the period of observation (first_revenue_date)
# or during the observation date (revenue).
# So, I'll create new variables that fill in the missing values with 0.


# In[5]:


#For revenue, , I create a new revenue column, called 'revenue2', which replaces NaN values with 0s.
indeed['revenue2']=indeed['revenue'].fillna(0)
indeed.describe()


# In[6]:


#For first revenue date, I'm less interested in the date of first revenue, and more 
# interested in which leads are nonmissing on this variable.
indeed['anyfirstrev']=indeed['first_revenue_date'].fillna(0)
indeed['anyfirstrev']
#So, I create an indicator for whether an advertising lead has ever given revenue.
# original 'first_revenue_date' variable, and filling in 0s for NaN values.


# In[7]:


#Next, I complete the indicator variable by substituting the date
# of first revenue for a value of 1 (meaning, yes there has been revenue
# before from this lead).
indeed['anyfirstrev'] = np.where(indeed['anyfirstrev']!=0, 1, 0)
indeed


# In[8]:


#Now that I've taken care of the missing data, I'll move onto the questions.

#QUESTION 1: How many leads are represented in this dataset?
#  To answer this question, I call the duplicated method on the feature
#  'advertiser_id' to see if there are any repeated values.
indeed.duplicated('advertiser_id')
ad_dups = indeed.advertiser_id[indeed.advertiser_id.duplicated()].values
ad_dups
# There are no duplicate entries for advertiser_id
#  so, there are 77,890 unique leads represented in this dataset.


# In[9]:


#QUESTION 1: How many leads are represented in this dataset?
#  To answer this question, I call the duplicated method on the feature
#  'advertiser_id' to see if there are any repeated values.
indeed.duplicated('advertiser_id')
ad_dups = indeed.advertiser_id[indeed.advertiser_id.duplicated()].values
ad_dups
# There are no duplicate entries for advertiser_id
#  so, there are 77,890 unique leads represented in this dataset.


# In[10]:


#QUESTION 1: Please describe both the assigned and unassigned populations.
#  To describe the unassigned and assigned populations, I call the groupby method
#  on the 'assigned' variable. The 'count' column tells me how many observations
#  are in each group.
indeed.groupby('assigned').describe()
# This shows that, of the ~77,000 leads, 40,812 are unassigned and 37,079 are assigned.


# In[11]:


indeed.groupby('assigned').mean()
# To further describe these two groups, I look at the means of each group.
# Compared to unassigned leads, the age of assigned leads is bigger, 
# the number of assign days is smaller, and revenue is higher.

#QUESTION 1: What is the average revenue of each group?
# Of assigned leads that have any revenue, the average revenue is $76,736,860.
#  Of all of the assigned leads (including those that do not yield revenue),
#  the average revenue is $3,238,846 (the value from the 'revenue2' column).

# Of unassigned leads that have any revenue, the average revenue is $23,889,420.
#  Of all of the unassigned leads (including those that do not yield revenue),
#  the average revenue is $1,039,001 (the value from the 'revenue2' column).


# In[12]:


# It looks like these values are quite different, but is this difference 
# statistically significant? To answer this question, I do a Chi2 test.
table = pd.crosstab(indeed.assigned, indeed.revenue)
chi2, p, dof, expected = chi2_contingency(table.values)
print('Chi-square Statistic %0.3f p_value %0.3f' % (chi2, p))
# The revenue of assigned and unassigned groups is significantly different.


# In[13]:


#QUESTION 2: What are the most important metrics to consider
# in this data set? Why?

# The key outcome variable in this dataset is revenue, and the
#  key explanatory variable is whether or not a lead has been assigned.
# From the prompt, these variables - and account age - seem to be
#   of greatest interest. However, I can't comment on which metrics
#   have the greatest explanatory power until after I've run my model.
#   I will return to this question below, after Question 4. 


# In[14]:


#QUESTION 3: Analyze any existing relationship between account age 
# and revenue, through visualization and other means.

# To begin, I plot age (x-axis) against revenue (y-axis)
get_ipython().magic('matplotlib inline')
indeed.plot(kind='scatter', x='age', y='revenue', c=['darkgray'], s=20)
plt.xlabel('Age of Account')
plt.ylabel('Revenue') 
plt.title('Relationship Between Account Age and Revenue')
plt.show()

#From this graph, we  can see that there's no easily discernable
# shape to the relationship between these two variables. Most of the
# revenue values cluster close to the bottom of the y-axis, regardless
# of the age of the account.


# In[15]:


#Another way to analyze this relationship would be to look at the 
# correlation between these two variables. I do this using the 
# Pearson R Correlation. 
pearsonr_coefficient, p_value = pearsonr(indeed.age, indeed.revenue2)
print('PearsonR Correlation Coefficient %0.3f' % (pearsonr_coefficient))
#I find that there is a weak positive relationship b/w age and revenue (0.02)


# In[16]:


#QUESTION 4: What is the incremental value from assigning a lead to the sales team?

#To answer this question, I'll run a linear regression, using the numerical variables
# in the dataset to predict revenue. 

#First, I create a trim dataset that includes only the numerical variables
indeed_trim = indeed[['assigned', 'age', 'assign_days', 'revenue2', 'anyfirstrev']]
indeed_trim


# In[17]:


#I'll explore this trim dataset by graphing a heatmap of correlations
# between all of the variables. 
X = indeed_trim
corr = X.corr()
corr
sb.heatmap(corr,xticklabels=corr.columns.values, yticklabels=corr.columns.values)
plt.show()
#It looks like the strongest correlation here between any two variables
# is between 'assigned' and 'age'.


# In[18]:


#Next, I prepare my X and y variables for modeling.
y = indeed_trim.revenue2
X = indeed_trim.drop(['revenue2'], axis=1)
#Revenue2 is the outcome variable, and the rest of the variables are X's.


# In[19]:


#Finally, I fit a linear regression to these data.
results = sm.OLS(y, X).fit()
results.summary()

# Whether or not a lead has been assigned is only marginally positively associated with revenue,
# meaning that assignment marginally increases revenue. But, the coefficient ($578,400)
# is quite big. Therefore, the incremental value of assigning a lead
# to a sales team is about $578,000 (but the relationship is only marginally significance).

#Other findings from this model:
# 1. Age does not significantly predict revenue.
# 2. The number of days assigned is negatively associated with revenue, meaning that 
#    the longer that a lead is assigned, the lower the revenue 
#    (Each additional day assigned is associated with a decline in revenue of $2,715)
# 3. Any first revenue (my dummy variable indicating whether the there was a non-
#    missing value on first_revenue) positively predicts revenue. This means that leads
#    who have previously given revenue have outcome revenues $19,100,000 higher than those
#    who have not previously given revenue.

# Returning to Question 2 (What metrics are most important to consider and why?), it looks
# as though my indicator variable of anyfirstrev is the most predictive of revenue. The
# number of assigned days is also significantly associated with revenue. Age was not a significant
# factor, and whether or not a lead was assigned was only marginally significant.

# One final thing to note is that the R2 of this model is quite low 
# (only 0.03% of variation is explained by this model).


# In[20]:


#BONUS QUESTION.Because the R2 of my linear regression model was so low,
# I thought a little bit about other models that I could run with these data.
# Because so many of the leads had a revenue value of 0, I was curious what 
# separated the leads with ANY revenue from those with none. 

#To answer this question, I first had to create a binary variable for whether
# or not there was any revenue during the period of observation.
indeed['anyrev'] = np.where(indeed['revenue2']!=0, 1, 0)


# In[21]:


#Next, I recreate the indeed_trim dataset, substituting 'anyrev' for 'revenue2'
indeed_trim = indeed[['assigned', 'age', 'assign_days', 'anyrev', 'anyfirstrev']]

#I also recreated the X and y to prepare for my logistic regression.
y = indeed_trim.anyrev
X = indeed_trim.drop(['anyrev'], axis=1)


# In[22]:


logit = sm.Logit(y, X)
result = logit.fit()
print(result.summary())


# In[23]:


print(np.exp(result.params))
# I exponentiate these coefficients for easier interpretation.

# The results of the logistic regression on ANY revenue look somewhat different
#  from the results of the linear regression I did in Question 4. 
#   1. First of all, being assigned to a salesperson actually REDUCES the odds of 
#    ANY revenue from that lead (by 86%).  
#   2. Having any previous revenue from that lead ('anyfirstrev') increases the odds of 
#    any revenue during the period of observation by a very large factor (~120x)
#   3. Finally, age and assign_days both reduce the odds of any revenue, 
#    but by small magnitudes (1% and 3% respectively)

# Also, the R2 of this logistic regression is much higher than that of the linear
# regression from Quetion 4. Here, the R2 is .38, meaning that 38% of the variance 
# in who does/does not pay out revenue is explained by this model. This model explains
# 35% more variance than the linear regression model. 

#To return to my undestanding of the situation above, it seems to me that the 
# prioritization scheme dictating which leads are and are not assigned is not 
# working terribly well. The only other interpretation would be that having a 
# sales rep working on a lead actually DETERS leads from paying - which seems unlikely!

