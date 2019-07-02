#!/usr/bin/env python
# coding: utf-8

# In[269]:


import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.metrics import r2_score
from sklearn import metrics
import pandas as pd
import pandas as pd
data = pd.read_csv(r'C:\Users\ngowt\OneDrive\Documents\SPRING 2019 Course work\Automatic controls of mechanical systems 512\project\DATA\DATA 11 states.csv')
data1= data.drop(999)
data1


# In[361]:


GMM = GaussianMixture(n_components=5).fit(data1) # Instantiate and fit the model
print('Converged:',GMM.converged_) # Check if the model has converged
means = GMM.means_ 
print('means')
print(means)
print(means.shape)
covariances = GMM.covariances_
print('covariances')
print(covariances)
print(covariances.shape)
weights = GMM.weights_
print('weights')
print(weights)
print(weights.shape)
# Predict
Y = data1
prediction = GMM.predict_proba(Y)
print(prediction)
print(prediction.shape)


# In[345]:


#Relevance
J = len(data1)
print(J)
s = data1.T
#print(s.shape)
list5 = []
for i in range(11):
    list5.append(weights)
    
#print(list5)    
mixing = np.array(list5)
print(mixing.shape)
#print(prediction.T.shape)
 
R = (1/J)*(np.matrix(prediction.T))*(np.matrix(data1))*(np.matrix(mixing))
print(R) 
Relevance = ([R.item(0,0)],[R.item(1,1)],[R.item(2,2)],[R.item(3,3)],[R.item(4,4)])
REL = np.array(Relevance)
print(REL)
m0=means[0]
m0


# In[346]:


#mN Eq=2.26 thesis
St_bar=np.mean(data1, axis=0)
St_bar=np.matrix(St_bar)
St_bar
print ('St_bar')
print (St_bar)
k0=1;
N=np.size(data1,axis=0);
print (N)
mN= ((k0/(k0*N))*m0) + ((N/(N+k0))*St_bar)
print ('mN')
print (mN)


# In[347]:


#kN Eq=2.27 thesis
k0=1;
N=np.size(data1,axis=0)
kN=k0+N;
kN


# In[348]:


#vN Eq=2.28 thesis
v0=1;
N=np.size(data1,axis=0)
vN=v0+N;
vN


# In[349]:


#Sst Eq 2.29 thesis
Sst=np.transpose(np.matrix(data1))*np.matrix(data1)
print(Sst)
Sst.shape


# In[350]:


# sN Eq= 2.30 thesis
k0=1
N=np.size(data1,axis=0)
t=(St_bar-m0)
T=np.transpose(t)
s0=covariances[0]
sN=s0+Sst+(k0*N/(k0+N))*(St_bar-m0)*T
print(sN)
sN.shape


# In[351]:


# Eq 2.31 
D=np.size(data1,axis=1)
Argmax=(sN/(kN+D+2))
print(Argmax)
Argmax.shape


# In[352]:



# Fxut 2.17
Ebb=Argmax[0:6,0:6]
Eba=Argmax[0:6,6:11]
Eab=Argmax[6:11,0:6]
Eaa=Argmax[6:11,6:11]
Fxut= Eab*np.linalg.inv(Ebb)
print (Fxut)
Fxut.shape


# In[353]:


# Fct Eq 2.18
mua= mN
mua=mN[0:1,6:11]
mua=np.transpose(mua)
mub=mN[0:1,0:6]
mub=np.transpose(mub)
Fct=mua-(Eab*np.linalg.inv(Ebb)*mub)
print (Fct)
Fct.shape


# In[354]:


# Ft Eq 2.19
Ft= Eaa-(Eab*np.linalg.inv(Ebb)*Eba)
print(Ft)
Ft.shape


# In[355]:


# X(t+1) Eq.4.20
data=np.matrix(data)
XU= data[0:999,0:6]
Xt=Fxut*np.transpose(XU)
QQ=np.repeat(Fct,repeats=999,axis=1)
QQ
y_pred=Xt+QQ
print(PP)            
y_pred.shape


# In[356]:


# checking the prediction
data1=np.matrix(data1)
y_true=data1[0:999,6:11]
y_true=np.transpose(y_true)
y_true


# In[357]:


# accuracy of prediction
r2_score(y_true,y_pred)


# In[ ]:





# In[ ]:




