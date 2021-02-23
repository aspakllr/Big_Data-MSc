# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 11:22:56 2019

@author: Schnitzel
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


# Import data 
beacons = pd.read_csv('beacons_final.csv', sep=';', index_col=False)
clinical = pd.read_csv('clinical_final.csv', sep=';', index_col=False)

# Merge clinical and beacons dataset together
mdata = pd.merge(beacons,clinical,on='part_id')
mdata = mdata.drop(['part_id'],axis=1)

mdata.to_csv('final.csv',sep=';',index=False)

# Categorization by fried
print(mdata.groupby('fried').size())
sns.set(rc={'figure.figsize':(11.7,8.27)})
fig = sns.countplot(mdata['fried'], palette="Set2").get_figure()
fig.savefig('original_fried_categorization.pdf')

# Prepare data for clustering
target = mdata.fried.values
# Keep features in a separate structure and remove their labels
features = mdata.drop(['fried'],axis=1)
# Convert to numparray
features = features.values

#Scaling of data
from sklearn.preprocessing import RobustScaler
ss = RobustScaler()
features = ss.fit_transform(features)

# ***************************** Clustering  *****************************
# K-means
from sklearn.cluster import KMeans

# Declaring Model
model = KMeans(n_clusters=3,algorithm='full')
# Fitting Model
model.fit(features)
# Prediction on the entire data
all_predictions = model.predict(features)
unique_elements, counts_elements = np.unique(all_predictions, return_counts=True)
print("K-means with 3 clusters:\n Clusters:", np.asarray(unique_elements), "contain:", np.asarray(counts_elements), "elements\n")
sns.countplot(all_predictions, palette="Set2").get_figure()

# Silhouette analysis
from sklearn.metrics import silhouette_samples, silhouette_score
score = silhouette_score (features, all_predictions, metric='euclidean')
print ("For 3 clusters silhouette score is:", score,"\n")

# Predict optimal number of clusters with Silhouette score
range_n_clusters = list (range(2,10))
print ("Silhouette analysis results: \n")
for n_clusters in range_n_clusters:
    clusterer = KMeans (n_clusters=n_clusters)
    preds = clusterer.fit_predict(features)
    centers = clusterer.cluster_centers_
    score = silhouette_score (features, preds, metric='euclidean')
    print ("For n_clusters = {}, silhouette score is {})".format(n_clusters, score),"\n")

# Elbow method   
sse = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000)
    kmeans = kmeans.fit(features)
    target = kmeans.labels_
    sse[k] = kmeans.inertia_ # Inertia: within-cluster sum-of-squares

plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of clusters k")
plt.ylabel("SSE error")
plt.title('Elbow method')
plt.show()
    

# **************** Dimensionality reduction  ****************
    
# PCA  
from sklearn.decomposition import PCA
pca = PCA()
pcompom = pca.fit_transform(features)
explained_variance = pca.explained_variance_ratio_  
plt.plot(explained_variance,'bo', color='firebrick')
plt.xlabel('Principal Components')
plt.ylabel('Variance')

# K-means clustering to reduced data
reduced_data = PCA(n_components=3).fit_transform(features)
kmeans = KMeans(n_clusters=3)
kmeans.fit(reduced_data)
all_predictions = kmeans.predict(reduced_data)
score = silhouette_score (reduced_data, all_predictions, metric='euclidean')
print("K-means with 3 clusters in reduced data:",score,"\n")

