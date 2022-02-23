#!/usr/bin/env python
# coding: utf-8

# In[10]:


#Finding Mean
from statistics import mean
# tuple of positive integer numbers
data1 = (11, 3, 4, 5, 7, 9, 2)
 
# tuple of a negative set of integers
data2 = (-1, -2, -4, -7, -12, -19)
 
# tuple of mixed range of numbers
data3 = (-1, -13, -6, 4, 5, 19, 9)
print("Mean of data set 1 is % s" % (mean(data1)))
print("Mean of data set 2 is % s" % (mean(data2)))
print("Mean of data set 3 is % s" % (mean(data3)))


# In[11]:


#Median
import statistics
 
# unsorted list of random integers
data1 = [2, -2, 3, 6, 9, 4, 5, -1]
 
 
# Printing median of the
# random data-set
print("Median of data-set is : % s "
        % (statistics.median(data1)))


# In[12]:


#Mode
import statistics
set1 =[1, 2, 3, 3, 4, 4, 4, 5, 5, 6]
print("Mode of given data set is % s" % (statistics.mode(set1)))


# In[13]:


#Measures of Variability
import statistics as st
nums=[1,2,3,5,7,9]
st.variance(nums)
st.pvariance(nums)
st.stdev(nums)


# In[14]:


#Linear Regression
import seaborn as sn
import matplotlib.pyplot as plt
sn.set(color_codes=True)
tips=sn.load_dataset('tips')
ax=sn.regplot(x='total_bill',y='tip',data=tips)
plt.show()


# In[17]:


import numpy as np
np.random.seed(7)
mean,cov=[3,5],[(1.3,.8),(.8,1.1)]
x,y=np.random.multivariate_normal(mean,cov,77).T
ax=sn.regplot(x=x,y=y,color='g')
plt.show()


# In[18]:


# Setting a Confidence Interval
ax=sn.regplot(x=x,y=y,ci=68)
plt.show()


# In[19]:


#The Chi-Square Test
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt 
x=np.linspace(0,10,100)
fig,ax=plt.subplots(1,1)
linestyles=['--','-.',':','-']
degrees_of_freedom=[1,3,7,5]
for df,ls in zip(degrees_of_freedom,linestyles):
     ax.plot(x,stats.chi2.pdf(x,df),linestyle=ls)
                   
plt.ylim(0,0.5)
plt.show()


# In[23]:


import numpy
import matplotlib.pyplot as plt

# number of sample
num = [1, 10, 50, 100]
# list of sample means
means = []

# Generating 1, 10, 30, 100 random numbers from -40 to 40
# taking their mean and appending it to list means.
for j in num:
	# Generating seed so that we can get same result
	# every time the loop is run...
	numpy.random.seed(1)
	x = [numpy.mean(
		numpy.random.randint(
			-40, 40, j)) for _i in range(1000)]
	means.append(x)
k = 0

# plotting all the means in one figure
fig, ax = plt.subplots(2, 2, figsize =(8, 8))
for i in range(0, 2):
	for j in range(0, 2):
		# Histogram for each x stored in means
		ax[i, j].hist(means[k], 10, density = True)
		ax[i, j].set_title(label = num[k])
		k = k + 1
plt.show()

