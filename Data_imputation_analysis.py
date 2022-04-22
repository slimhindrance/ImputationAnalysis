#!/usr/bin/env python
# coding: utf-8

# In[35]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression
hw = pd.read_csv('data.csv')
hw['bmi'] = hw.Weight/(hw.Height**2)


# In[36]:


weight = [63.88,60.67,60.1,55.21,53.63,53,52,51.92,51.87,51.8,51.58,51.45,51.27,51.15,51.13,51.11]
bmi = [29.56,28.08,27.81,25.55,24.82,24.53,24.06,24.03,24,23.97,23.87,23.81,23.72,23.67,23.66,23.65]
iterations = [3401,14401,18001,55201,84001,105201,178701,190901,201101,218001,273401,318001,418001,518001,618001,718001]
plt.title("Bayesian BMI Estimate")
plt.ylabel("BMI")
plt.xlabel("Iterations")
plt.vlines(x=178701,ymin=22,ymax=30, color='red')
plt.hlines(y=24.06, xmin=0,xmax=178701, color='red')
plt.plot(iterations, bmi)


# In[37]:


plt.title("Height on Weight")
plt.xlabel("Height (m)")
plt.ylabel("Weight (kg)")
plt.plot(hw.Height,hw.Weight)
hw


# In[38]:


import seaborn as sns
plt.title('Distribution of Weight')
#plt.xticks([1.3,1.4,1.5,1.6,1.7,1.8,1.9,2])
plt.legend(
)
sns.kdeplot(hw['Weight'])
plt.vlines(x=np.mean(hw.Weight), ymin=0, ymax=.06)
sns.kdeplot(hw['Weight'][1:])
plt.vlines(x=np.mean(hw.Weight[1:]), ymin=0, ymax=.06, color='orange')
sns.kdeplot(hw['Weight'][:-1])
plt.vlines(x=np.mean(hw.Weight[:-1]), ymin=0, ymax=.06, color='green')


# In[39]:


plt.title('Distribution of Weight - Omitting 3')
#plt.xticks([1.3,1.4,1.5,1.6,1.7,1.8,1.9,2])
plt.legend(
)
sns.kdeplot(hw['Weight'])
plt.vlines(x=np.mean(hw.Weight), ymin=0, ymax=.06)
sns.kdeplot(hw['Weight'][3:])
plt.vlines(x=np.mean(hw.Weight[3:]), ymin=0, ymax=.06, color='orange')
sns.kdeplot(hw['Weight'][:-3])
plt.vlines(x=np.mean(hw.Weight[:-3]), ymin=0, ymax=.06, color='green')


# In[40]:


kw = hw.drop([0,2,4,6,8,10,12,14])
plt.title('Distribution of Weight - Omitting ')
#plt.xticks([1.3,1.4,1.5,1.6,1.7,1.8,1.9,2])
plt.legend(
)
sns.kdeplot(kw['Weight'])
plt.vlines(x=np.mean(kw.Weight), ymin=0, ymax=.06)
sns.kdeplot(hw.Weight)


# In[41]:


lmhw = hw.copy()
hmhw = hw.copy()
lmhw.Weight[:3] = np.mean(lmhw.Weight[3:])
hmhw.Weight[-3:] = np.mean(hmhw.Weight[:-3])


# In[ ]:





# In[42]:



plt.title("Mean-imputation of 3 Edge Cases")
sns.kdeplot(hw.Weight)
plt.vlines(x=np.mean(hw.Weight), ymin=0, ymax=.06)
sns.kdeplot(lmhw.Weight)
plt.vlines(x=np.mean(lmhw.Weight), ymin=0, ymax=.06, color='orange')
sns.kdeplot(hmhw.Weight)
plt.vlines(x=np.mean(hmhw.Weight), ymin=0, ymax=.06, color='green')


# In[43]:


"""\begin{marginfigure}%
  \includegraphics[width=\linewidth]{hwmean.png}
  \caption{Distributions of the data using mean-imputation of missing data for the least and greatest weights.}
  \label{fig:marginfig}
\end{marginfigure}"""


# In[44]:


#hot deck
hhdhw = hw.copy()
lhdhw = hw.copy()
hhdhw.Weight[:3] = hw.Weight[3]
lhdhw.Weight[-3:] = hw.Weight[11]


plt.title("Hot Deck Imputation of 3 Edge Cases")
sns.kdeplot(hw.Weight)
plt.vlines(x=np.mean(hw.Weight), ymin=0, ymax=.08)
sns.kdeplot(lmhw.Weight)
plt.vlines(x=np.mean(hhdhw.Weight), ymin=0, ymax=.08, color='orange')
sns.kdeplot(hmhw.Weight)
plt.vlines(x=np.mean(lhdhw.Weight), ymin=0, ymax=.08, color='green')


# In[45]:


mhdhw = hw.copy()
mhdhw.Weight[4:7] = hw.Weight[3]
mhdhw.Weight[7:9] = hw.Weight[9]
mhdhw
plt.title("Hot Deck Imputation of 5 Central Values")
sns.kdeplot(hw.Weight)
plt.vlines(x=np.mean(hw.Weight), ymin=0, ymax=.05)
sns.kdeplot(mhdhw.Weight)
plt.vlines(x=np.mean(mhdhw.Weight), ymin=0, ymax=.05, color='orange')


# In[ ]:





# ### Building a linear regression model

# In[71]:


bases=[]
omits=[]
means=[]
medians=[]
gibbss=[]
for i in range(50):
    #BASELINE WITH ALL DATA (Expecting Best possible)
    X_train, X_test, y_train, y_test = tts(hw.Height,hw.Weight, test_size=.2, random_state=i)
    X_train= np.array(X_train).reshape(-1, 1)
    y_train= np.array(y_train).reshape(-1, 1)
    X_test = np.array(X_test).reshape(-1, 1)
    clf = LinearRegression()
    base = clf.fit(X_train, y_train).score(X_test,y_test)
    #OMIT DATUM
    omit =clf.fit(X_train[1:], y_train[1:]).score(X_test,y_test)
    # impute with mean
    hw.Weight[0]=np.mean(hw.Weight)
    X_train, X_test, y_train, y_test = tts(hw.Height,hw.Weight, test_size=.2, random_state=i)
    X_train= np.array(X_train).reshape(-1, 1)
    y_train= np.array(y_train).reshape(-1, 1)
    X_test = np.array(X_test).reshape(-1, 1)
    clf = LinearRegression()
    mean = clf.fit(X_train, y_train).score(X_test,y_test)
    #median
    hw.Weight[0]= np.median(hw.Weight)
    X_train, X_test, y_train, y_test = tts(hw.Height,hw.Weight, test_size=.2, random_state=i)
    X_train= np.array(X_train).reshape(-1, 1)
    y_train= np.array(y_train).reshape(-1, 1)
    X_test = np.array(X_test).reshape(-1, 1)
    clf = LinearRegression()
    median = clf.fit(X_train, y_train).score(X_test,y_test)
    
                   
    
results = pd.DataFrame()
results['base']=bases
results['omit']=omits
results['mean']=means
results['median']=medians


# In[72]:


results


# In[73]:


#min(results.gibbs_mean_diff), np.mean(results.gibbs_mean_diff), max(results.gibbs_mean_diff), np.mean(results)


# Unexpectedly, omitting the variable performs better than the base case with all original variables.  However, we definitely see a far better accuracy for the model with the GIBBS estimate compared with either mean or median imputation.
# 
# Let's try the other extreme case $\rightarrow$ the heaviest weight.

# In[74]:


#Finding the best estimate of Weight[-1]

weight = [70.41,71.38,71.85,72.07,72.23,72.26]
bmi = [21.03,21.31,21.45,21.52,21.57,21.58]
iterations = [28001,58001,88001,118001,148001,178001]
plt.title("Bayesian BMI Estimate")
plt.ylabel("BMI")
plt.xlabel("Iterations")
plt.vlines(x=148001,ymin=min(bmi),ymax=max(bmi), color='red')
plt.hlines(y=bmi[4], xmin=min(iterations),xmax=148001, color='red')
plt.plot(iterations, bmi), 'weight estimated as: ' + str(weight[4])


# In[75]:


bases=[]
omits=[]
means=[]
medians=[]
gibbss=[]
for i in range(50):

    #OMIT DATUM
    omit =clf.fit(X_train[:-1], y_train[:-1]).score(X_test,y_test)
    # impute with mean
    hw.Weight[14]=np.mean(hw.Weight)
    X_train, X_test, y_train, y_test = tts(hw.Height,hw.Weight, test_size=.2, random_state=i)
    X_train= np.array(X_train).reshape(-1, 1)
    y_train= np.array(y_train).reshape(-1, 1)
    X_test = np.array(X_test).reshape(-1, 1)
    clf = LinearRegression()
    mean = clf.fit(X_train, y_train).score(X_test,y_test)
    #median
    hw.Weight[14]= np.median(hw.Weight)
    X_train, X_test, y_train, y_test = tts(hw.Height,hw.Weight, test_size=.2, random_state=i)
    X_train= np.array(X_train).reshape(-1, 1)
    y_train= np.array(y_train).reshape(-1, 1)
    X_test = np.array(X_test).reshape(-1, 1)
    clf = LinearRegression()
    median = clf.fit(X_train, y_train).score(X_test,y_test)
    #Impute with GIBBS Estimate
    hw.Weight[14]= 71.83 #GIBBS estimate
    X_train, X_test, y_train, y_test = tts(hw.Height,hw.Weight, test_size=.2, random_state=i)
    X_train= np.array(X_train).reshape(-1, 1)
    y_train= np.array(y_train).reshape(-1, 1)
    X_test = np.array(X_test).reshape(-1, 1)
    clf = LinearRegression()
    gibbs = clf.fit(X_train, y_train).score(X_test,y_test)
    
    bases.append(max(0,base))
    omits.append(max(0,omit))
    means.append(max(0,mean))
    medians.append(max(0,median))
    gibbss.append(max(0,gibbs))

    
results = pd.DataFrame()
results['base']=bases
results['omit']=omits
results['mean']=means
results['median']=medians
results['gibbs']=gibbss
    
results['gibbs_mean'] = [True if results.gibbs[x]>results['mean'][x] else False for x in range(len(results))]
#results['gibbs_median'] = [True if results.gibbs[x]>results['median'][x] else False for x in range(len(results))]
#results['gibbs_omit'] = [True if results.gibbs[x]>=results['omit'][x] else False for x in range(len(results))]
results['gibbs_mean_diff'] = [results.gibbs[x]-results['mean'][x] for x in range(len(results))]
results['as_good'] = [True if results.gibbs[x]-results.base[x]==0 else False for x in range(len(results))]
results.head()


# In[76]:


min(results.gibbs_mean_diff), np.mean(results.gibbs_mean_diff), max(results.gibbs_mean_diff), np.mean(results)


# # Results
# Here we see that the GIBBS estimate outperforms all all other methods.
# 
# This was a simple toy example with a small dataset, with pretty small stakes.  Let's see what happens when we're looking at something a bit more complicated.
# 
# 

# # Multivariate Mess MNAR
# Here we test several datasets to compare the effectiveness of GIBBS sampling for data imputation compared with omitting the value, or substituting with either mean, median or mode.

# In[77]:


df = pd.read_csv('Fish.csv')


# In[78]:


#MNAR omission
np.random.seed(42) #for reproducibility
for x in df[df.Species == 'Smelt'].index:
    df.Length1[x] = 'NA'
for x in df[df.Species == 'Bream'].index:
    df.Length2[x] = 'NA'
for x in df[df.Species == 'Roach'].index:
    df.Length3[x] = 'NA'
for x in df[df.Species == 'Parkki'].index:
    df.Height[x] = 'NA'
for x in df[df.Species == 'Perch'].index:
    df.Width[x] = 'NA'
    
X,x = tts(df, test_size=.1, stratify=df.Species)
x.sort_values(by='Species')

gdf = df.copy()

L1fill = [8.797,9.5,9.5,9.8,10.2,10.4,10.8,10.8,10.9,11.1,11.2,11.5,12.7,16.6]
L2fill = [28.18,29.336,29.24,31.49,31.96,32.65,32.49,32.89,33.03,34.07,34.07,34.08,34.24,35.05,35.04,35.06,36.08,36.27,36.37,36.43,37.18,36.95,37.39,38.23,38.17,38.55,38.27,39.11,39.24,40.26,41.61,41.52,42.75,43.31,43.84]
L3fill = [15.61,19.56,20.7,21.5,21.9,22.3,22.45,22.8,23.93,24.03,24.01,24.58,24.72,25.67,25.71,25.84,27,47,27.95,29.04,34.07]
Hfill = [5.966,5.978,6.079,6.225,6.387,6.27,6.5,6.3,6.6,7,7.15]
Wfill = [2.3,2.69,2.787,2.95,2.79,2.95,2.99,3.13,3.13,3.19,3.16,3.16,3.3,3.3,3.3,3.43,3.41,3.345,3.423,3.397,3.315,3.4,3.55,3.44,3.36,3.55,3.78,3.55,3.67,3.63,3.69,3.85,4.123,3.9,3.9,3.92,4.008,4.08,4.141,4.818,4.95,5.5,5.28,2.26,5.3,5.26,5.68,5.35,5.75,5.74,5.78,6,5.6,6.262,6,6.26,6.143,6.154]

l1=[]
l2=[]
l3=[]
h=[]
w=[]
for i in range(len(gdf)):
    if gdf.Length1[i] != 'NA':
        l1.append(gdf.Length1[i])
        
    else:
        l1.append(L1fill[0])
        L1fill.pop(0)
    if gdf.Length2[i] != 'NA':
        l2.append(gdf.Length2[i])
    else:
        l2.append(L2fill[0])
        L2fill.pop(0)
    if gdf.Length3[i] != 'NA':
        l3.append(gdf.Length3[i])
    else:
        l3.append(L3fill[0])
        L3fill.pop(0)
    if gdf.Height[i] != 'NA':
        h.append(gdf.Height[i])
    else:
        h.append(Hfill[0])
        Hfill.pop(0)
    if gdf.Width[i] != 'NA':
        w.append(gdf.Width[i])
    else:
        w.append(Wfill[0])
        Wfill.pop(0)
gdf.Length1 = l1
gdf.Length2 = l2
gdf.Length3 = l3
gdf.Height = h
gdf.Width = w


# In[79]:


df=df.replace('NA',np.nan)


# In[80]:


#FILL NAs with mean of column
meandf=df.drop('Species',axis=1).apply(lambda x: x.fillna(x.mean()),axis=0)
meandf['Species'] = df.Species
#TOTAL ABSOLUTE DIFFERENCE BETWEEN ORIGINAL DATASET AND MEAN IMPUTED ONE
 
abs(meandf.drop('Species',axis=1)- pd.read_csv("Fish.csv").drop('Species',axis=1)).values.sum()


# In[81]:


abs(gdf.drop('Species',axis=1)- pd.read_csv("Fish.csv").drop('Species',axis=1)).values.sum()


# In[82]:


from sklearn.metrics import r2_score
#PREP FOR LINEAR REGRESSION
for col in pd.get_dummies(data=df, drop_first=True).columns:
    if col not in gdf.columns:
        gdf[col]=pd.get_dummies(data=df, drop_first=True)[col]
    if col not in meandf.columns:
        meandf[col]=pd.get_dummies(data=df, drop_first=True)[col]


# # Regression

# In[83]:


from sklearn.metrics import mean_squared_error as MSE


# In[84]:


#Mean Imputation regression
mX_train, mX_test, my_train, my_test = tts(meandf.drop(['Species','Weight'], axis=1), meandf['Weight'], test_size=.3, random_state=4)
clf = LinearRegression()
clf.fit(mX_train, my_train).score(mX_train,my_train)
MSE(clf.predict(mX_test),my_test)


# In[85]:


gX_train, gX_test, gy_train, gy_test = tts(gdf.drop(['Species','Weight'], axis=1), meandf['Weight'], test_size=.3, random_state=4)
clf = LinearRegression()
clf.fit(gX_train, gy_train).score(gX_train,gy_train)
MSE(clf.predict(gX_test),gy_test)


# In[86]:


from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

mregr = OLS(my_test, add_constant(mX_test)).fit()
gregr = OLS(gy_test, add_constant(gX_test)).fit()
print(mregr.bic, gregr.bic)


# In[87]:


#GIBBS estimate
X_train, X_test, y_train, y_test = tts(gdf.drop(['Species','Weight'],axis=1), df['Weight'], test_size=.2, random_state=42)
clf = LinearRegression()
clf.fit(X_train, y_train).score(X_test,y_test)


# In[88]:


bases=[]
means=[]
gibbss=[]
#clf=LinearRegression()
for size in [.15,.2,.25,.3]:
    for i in range(100):


        # impute with mean
        X_train, X_test, y_train, y_test = tts(meandf.drop(['Species', 'Weight'],axis=1),df.Weight, test_size=size, random_state=i)
        #X_train= np.array(X_train).reshape(-1, 1)
        #y_train= np.array(y_train).reshape(-1, 1)
        #X_test = np.array(X_test).reshape(-1, 1)
        mean = clf.fit(X_train, y_train).score(X_test,y_test)
        #Impute with GIBBS Estimate
        X_train, X_test, y_train, y_test = tts(gdf.drop(['Species','Weight'], axis=1),df.Weight, test_size=size, random_state=i)
        #X_train= np.array(X_train).reshape(-1, 1)
        #y_train= np.array(y_train).reshape(-1, 1)
        #X_test = np.array(X_test).reshape(-1, 1)
        clf = LinearRegression()
        gibbs = clf.fit(X_train, y_train).score(X_test,y_test)



        means.append(max(0,mean))

        gibbss.append(max(0,gibbs))

    
results = pd.DataFrame()
results['base']=bases
results['mean']=means
results['gibbs']=gibbss
    
results['gibbs_mean'] = [True if results.gibbs[x]>results['mean'][x] else False for x in range(len(results))]
#results['gibbs_median'] = [True if results.gibbs[x]>results['median'][x] else False for x in range(len(results))]
#results['gibbs_omit'] = [True if results.gibbs[x]>=results['omit'][x] else False for x in range(len(results))]
results['gibbs_mean_diff'] = [results.gibbs[x]-results['mean'][x] for x in range(len(results))]
results['as_good'] = [True if results.gibbs[x]-results.base[x]==0 else False for x in range(len(results))]
results.head()
np.mean(results)


# In[89]:



sum([1 if x==True else 0 for x in results.gibbs_mean])/len(results)


# In[90]:


results


# In[98]:


#sns.pairplot(pd.read_csv("Fish.csv"))
sns.pairplot(pd.read_csv("Fish.csv"), hue="Species")


# In[95]:





# In[ ]:




