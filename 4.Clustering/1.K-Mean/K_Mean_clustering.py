  import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Mall_Customers.csv')
x=dataset.iloc[:,3:].values 


#elbow method

from sklearn.cluster import KMeans

wcss=[] #list
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow method')
plt.xlabel('no of clusters')
plt.ylabel('wcss')
plt.show()

# from sklearn.datasets import make_blobs

# from yellowbrick.cluster import KElbowVisualizer

kmeans=KMeans(n_clusters =5, init='k-means++',random_state =42)
y_kmeans=kmeans.fit_predict(x)

print(y_kmeans)

plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=100,c='red',label='cluster 1')

plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=100,c='Blue',label='cluster 2')


plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=100,c='green',label='cluster 3')


plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],s=100,c='cyan',label='cluster 4')

plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1],s=100,c='Magenta',label='cluster 5')

plt.scatter(kmeans.cluster_centers_[:,0],KMeans.cluster_centers_[:,1],s=300,c='orange',label='Centroids',alpha=0.5)

plt.title('K-Means clustering (clusters of customers)')

plt.xlabel('Annual Income(K$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
