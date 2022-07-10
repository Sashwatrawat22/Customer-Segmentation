#!/usr/bin/env python
# coding: utf-8
Importing libraries
# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans


# In[2]:


# loading the data from csv file to a Pandas DataFrame
cust_data = pd.read_csv('Mall_Customers.csv')

# first 5 rows in the dataframe
cust_data.head()


# In[3]:


# finding the number of rows and columns
cust_data.shape


# In[4]:


# getting some informations about the dataset
cust_data.info()

#Non-Null Count- tells us that there is no null values in our dataset


# In[5]:


cust_data.describe()

Countplot of Gender Distribution
# In[28]:


plt.figure(figsize = (12 , 6))
sns.countplot(y = 'Gender' , data = cust_data)
plt.show()


# In[ ]:


1. Choosing the Annual Income Column & Spending Score 


# In[6]:


X = cust_data.iloc[:,3:5]
X


Finding the optimum no. of clusters
WCSS(Within Sum Of Squares)

# In[7]:


wcss = []

for i in range(1,9):
  kmeans = KMeans(i)
  kmeans.fit(X)
  wcss.append(kmeans.inertia_)

Finding the no. of clusters using Elbow Method
# In[8]:


plt.plot(range(1,9),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Within Cluster Sum of Squares')
plt.show()

#This shows us that the elbow point is at n=5, so 5 is the optimum value of the no. of clusters

Training the k-Means Clustering Model
# In[9]:


kmeans = KMeans(5)

# return a label for each data point based on their cluster
Y = kmeans.fit_predict(X)
Y


# In[10]:


clusters=X.copy()
clusters['clusters_Pred']=Y
clusters


# In[11]:


# plot the Clusters and their respective Centroids

plt.figure(figsize=(8,6))
plt.scatter(clusters['Annual Income (k$)'],clusters['Spending Score (1-100)'],c=clusters['clusters_Pred'],cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='black', label='Centroids')

plt.legend()
plt.title('Customer Groups')
plt.xlabel('Annual Income (k$) ')
plt.ylabel('Score (1-100)')
plt.show()

1. Choosing the Age & Spending Score 
# In[12]:


X1 = cust_data.iloc[:,2:5:2]
X1

Finding the optimum no. of clusters using WCSS(Within Sum Of Squares)
# In[21]:


wcss = []

for i in range(1,10):
  kmeans = KMeans(i)
  kmeans.fit(X1)
  wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,6))
plt.plot(range(1,10),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Within Cluster Sum of Squares')
plt.show()

#This shows us that the elbow point is at n=4, so 4 is the optimum value of the no. of clusters


# In[22]:


kmeans = KMeans(4)

# return a label for each data point based on their cluster
Y1 = kmeans.fit_predict(X1)
Y1


# In[23]:


clusters2=X1.copy()
clusters2['clusters_Pred']=Y1
clusters2


# In[24]:


# plot the Clusters and their respective Centroids

plt.figure(figsize=(8,6))
plt.scatter(clusters2['Age'],clusters2['Spending Score (1-100)'],c=clusters2['clusters_Pred'],cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='black', label='Centroids')

plt.legend()
plt.title('Customer Groups')
plt.xlabel('Age ')
plt.ylabel('Score (1-100)')
plt.show()


# In[ ]:




