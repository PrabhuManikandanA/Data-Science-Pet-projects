import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Mall_Customers.csv')
x=dataset.iloc[:,3:].values 


#Dendrogram

import scipy.cluster.hierarchy as sch
dendrogram =sch.dendrogram(sch.linkage(x,method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Eculidean distances')
plt.show()

#training the Heirarchical clustering model on the dataset 

from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
y_hc=hc.fit_predict(x)

print(y_hc)

#Visualisation the cluster 


plt.scatter(x[y_hc==0,0],x[y_hc==0,1],s=100,c='red',label='cluster 1')

plt.scatter(x[y_hc==1,0],x[y_hc==1,1],s=100,c='Blue',label='cluster 2')


plt.scatter(x[y_hc==2,0],x[y_hc==2,1],s=100,c='green',label='cluster 3')


# plt.scatter(x[y_hc==3,0],x[y_hc==3,1],s=100,c='cyan',label='cluster 4')

# plt.scatter(x[y_hc==4,0],x[y_hc==4,1],s=100,c='Magenta',label='cluster 5')

# plt.scatter(kmeans.cluster_centers_[:,0],KMeans.cluster_centers_[:,1],s=300,c='orange',label='Centroids',alpha=0.5)

plt.title('K-Means clustering (clusters of customers)')

plt.xlabel('Annual Income(K$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
