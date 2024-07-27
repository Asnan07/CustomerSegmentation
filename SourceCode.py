import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

customerdata = pd.read_csv('Mall_Customers.csv')

customerdata.head()

customerdata.shape

customerdata.info()

#check for missing values
customerdata.isnull().sum()

#choosing annual income column and spending score column
X = customerdata.iloc[:,[3,4]].values

print(X)

#Choosing optimun no of clusters by using parameter
#WCSS --> WITHIN CLUSTERS SUM OF SQUARES
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init ='k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


#plot an elbow graph
sns.set()
plt.plot(range(1,11), wcss)
plt.title('The Elbow Point Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
#optimum no of cluster is 5 as shown in above graph

#training the kmeans clustering model

kmeans = KMeans(n_clusters=5, init ='k-means++', random_state=0)

#return a label for each data point based on their clusters
Y = kmeans.fit_predict(X)
print(Y)

#visualizing all the clusters

#plotting all clusters and their centroids

plt.figure(figsize=(8,8))
plt.scatter(X[Y==0,0], X[Y==0,1], s = 50, c ='green', label='Cluster 1')
plt.scatter(X[Y==1,0], X[Y==1,1], s = 50, c ='red', label='Cluster 2')
plt.scatter(X[Y==2,0], X[Y==2,1], s = 50, c ='yellow', label='Cluster 3')
plt.scatter(X[Y==3,0], X[Y==3,1], s = 50, c ='blue', label='Cluster 4')
plt.scatter(X[Y==4,0], X[Y==4,1], s = 50, c ='purple', label='Cluster 5')

#plot centroids
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 100, c ='cyan', label = 'Centroid')
plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()
