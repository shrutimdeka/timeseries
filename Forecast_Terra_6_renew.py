#!/usr/bin/env python
# coding: utf-8

# In[23]:


##################### Start with simple to complex models ##########
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import itertools
import statsmodels.api as sm


# In[2]:


########################### load training data #################
train = pd.read_csv("C:/Users/Shruti/Downloads/train_csv.csv")
#timestamp index
train.index = pd.to_datetime(train.time)
########################### View nature of data ###############
train.feature.plot()


# In[3]:


train.head(10) #every 10 seconds data is recorded


# In[4]:


################### Test of stationarity ############

#DF Test - Alt Hyp = TREND stationary
test1 = adfuller(train.feature)
print("p-value: ", test1[1])
print("lags: ", test1[2])
print("test statistics: ", test1[0])
print("critical values: ", test1[4])

#p >> all critical values
#Null hypothesis is accepted = Series is NOT stationary


# In[15]:


#################### Create stationarity ###########
#1)differencing - trend removal

lag= 1
diff=[]
for i in range(len(train.feature)):
    diff.append(train.feature[i] - train.feature[i-lag]) 
plt.plot(diff)


# In[ ]:


diff  #negative values are present


# In[20]:


#2)Log transformation - remove trend

new_feature = []
for i in range(train.shape[0]):
    new_feature.append(np.log(train.feature[i])) #Nope  XXXXXXX
plt.plot(new_feature)                  #trend doesn't oscillate around a constant 

#No improvement - let's go with single differencing transformation 


# In[ ]:


########## Use differenced data for SARIMA and see if it is better than ###########
##################  SARIMA without any transformation   #############################


# In[16]:


################# Use DF Test to see if differencing worked #####
test1 = adfuller(diff)
print("p-value: ", test1[1])
print("lags: ", test1[2])
print("test statistics: ", test1[0])
print("critical values: ", test1[4]) #Still NOT stationary- p value >> critical values

#USE SARIMA on original data


# In[22]:


########################## ACF & PACF plot to find periods #############

plot_acf(train.feature)
plt.show()    # 1 lag
plot_pacf(train.feature)
plt.show()    # 1 lag 


# In[31]:


###################### Seasonal Decompose ####################
decom = seasonal_decompose(train.feature, freq=12) # separate trend, seasonal components in data
decom.plot()  #freq = 12 as is clear in data plot

#residual does seem to have a little pattern, but overall it is a good decomposition.


# In[32]:


####################### choose BEST SARIMA parameter for model #############
p = q = d = range(0, 2)
pdq = [x for x in itertools.product(p, d, q)]

seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
seasonal_pdq   #SEASONAL: AR, diff, MA, M= freq of 1 seasonal period

res = []
for x in pdq:
    for x_seasonal in seasonal_pdq:
        model = sm.tsa.statespace.SARIMAX(train.feature,
                                        order=x,
                                        seasonal_order=x_seasonal,
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
        model = model.fit()
        m = [x, x_seasonal, model.aic]   #put model parameters and its aic into a list
        res.append(m)

res_df = pd.DataFrame(res, columns = ['order', 'seasonal_order', 'aic'])

res_df[res_df['aic'] == min(res_df.aic)]  #display parameters for min AIC


# In[33]:


############# SARIMA to model seasonality + trend after parameter choosing ###########
mod = sm.tsa.statespace.SARIMAX(train.feature,
                                order=(1, 1, 1),
                                seasonal_order=(0,1, 1, 12),
                                enforce_stationarity=True,
                                enforce_invertibility=False)
results = mod.fit()


# In[34]:


results.fittedvalues  #see fitted values by the model on training data


# In[35]:


###################### Load Test data ###############
test = pd.read_csv("C:/Users/Shruti/Downloads/test_csv.csv")
test.index = pd.to_datetime(test['time'])


# In[36]:


######################## Prediction usig model #############
pred = results.predict(start= test.index[0], end = test.index[-1], dynamic= False)


# In[43]:


############################ plot the predictions and past values ###############################
plt.figure(figsize=(10, 8))
plt.plot(train['feature'], color='blue', label="train data")     #original train data
plt.plot(results.fittedvalues, color='green', label="fitted values") #fitted values
plt.plot(pred, color='red', label="predicted values")                   #predicted values
plt.legend()
plt.show()


# In[44]:


############ See Model Residual Autocorrelation #######

residuals = [x for x in (train.feature - results.fittedvalues)]
residual_df = pd.DataFrame(residuals)

plot_acf(residual_df)
plt.show() #No autocorrelation

plot_pacf(residual_df)
plt.show() #No autocorrelation


# In[45]:


#see distribution of residuals for bias (should be around mean zero)
plt.hist(residuals)
#skewed - some bias exists


# In[ ]:


##### save into csv for submission ####
pred_df = pd.DataFrame(test.id, columns=['id'])

pred_df['feature'] = pred.values

pred_df.to_csv('terra_timeseries_sarimax_stationarity.csv') #115 score
#####################################################################

