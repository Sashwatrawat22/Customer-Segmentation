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

import warnings         #to ignore unnecessary warnings
warnings.filterwarnings('ignore')


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


# In[6]:


#Plotting age vs annual income 
plt.figure(figsize=(10,6))
plt.scatter(cust_data['Age'],cust_data['Annual Income (k$)'],alpha=0.4,c='blue')
plt.title("Age vs Annual Income ")
plt.ylabel('Annual Income (k$)')
plt.xlabel("Age")
plt.show()

#most of the customers have annual income in the range of (40-80)K$

Countplot of Gender Distribution
# In[7]:


plt.figure(figsize=(12,6))
sns.countplot(x='Gender', data=cust_data)
plt.show()
#This shows us that there are more female customers than male customers.

 | Choosing the Annual Income Column & Spending Score for Cluster Analysis
# In[8]:


X = cust_data.iloc[:,3:5]
X


Finding the optimum no. of clusters
WCSS(Within Sum Of Squares)

# In[9]:


wcss = []

for i in range(1,9):
  kmeans = KMeans(i)
  kmeans.fit(X)
  wcss.append(kmeans.inertia_)

Finding the no. of clusters using Elbow Method
# In[10]:


plt.plot(range(1,9),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Within Cluster Sum of Squares')
plt.show()

#This shows us that the elbow point is at n=5, so 5 is the optimum value of the no. of clusters

Training the k-Means Clustering Model
# In[11]:


kmeans = KMeans(5)

# return a label for each data point based on their cluster
Y = kmeans.fit_predict(X)
Y
#5 labels are created for 5 clusters same label for same cluster


# In[12]:


clusters=X.copy()
clusters['clusters_Pred']=Y
clusters


# In[13]:


# plot the Clusters and their respective Centroids

plt.figure(figsize=(8,6))
plt.scatter(clusters['Annual Income (k$)'],clusters['Spending Score (1-100)'],c=clusters['clusters_Pred'],cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='black', label='Centroids')

plt.legend()
plt.title('Customer Groups')
plt.xlabel('Annual Income (k$) ')
plt.ylabel('Score (1-100)')
plt.show()

Inference:

Red color Cluster   - earning less , spending more
Green Color Cluster - average in terms of earning and spending 
Blue Color Cluster  - earning high and also spending high 
Purple Color Cluster- earning less and spending less
Orange Color Cluster- earning high but spending less

-All the customers that come in the blue cluster should be the main target in order to the business.
-Feedback needs to be taken so that the mall can improve those areas in order to convert the customers in orange cluster to 
the blue cluster.
|| Choosing the Age & Spending Score 
# In[14]:


X1 = cust_data.iloc[:,2:5:2]
X1

Finding the optimum no. of clusters using WCSS(Within Sum Of Squares)
# In[15]:


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


# In[16]:


kmeans = KMeans(4)

# return a label for each data point based on their cluster
Y1 = kmeans.fit_predict(X1)
Y1
#as 4 cluster are to be made so only 4 labels are generated


# In[17]:


clusters2=X1.copy()
clusters2['clusters_Pred']=Y1
clusters2


# In[18]:


# plot the Clusters and their respective Centroids

plt.figure(figsize=(8,6))
plt.scatter(clusters2['Age'],clusters2['Spending Score (1-100)'],c=clusters2['clusters_Pred'],cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='black', label='Centroids')

plt.legend()
plt.title('Customer Groups')
plt.xlabel('Age ')
plt.ylabel('Score (1-100)')
plt.show()

Inference:

Red color Cluster   - old customers, average spending
Green Color Cluster - average in terms of earning and spending 
cyan Color Cluster  - young customers with average spending 
Purple Color Cluster- young customers with high spending 
Pale Yellow Color Cluster- covers almost every age group customers with less spending
