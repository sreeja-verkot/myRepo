#https://www.simplilearn.com/unsupervised-learning-with-clustering-machine-learning-tutorial
#https://www.kaggle.com/karthickaravindan/k-means-clustering-project
#https://www.kaggle.com/sashr07/unsupervised-learning-tutorial

import matplotlib.pyplot as plt
from sklearn import datasets
import sklearn.metrics as sm
 
import pandas as pd
import numpy as np
 
# Only needed if you want to display your plots inline if using Notebook
# change inline to auto if you have Spyder installed
# import some data to play with
iris = datasets.load_iris()
print(iris.target_names)
species_dict = dict(zip(range(0, len(iris.target_names)), iris.target_names))
print(species_dict)
print(iris.target)
iris_species = list((map(lambda x : species_dict[x], iris.target)))
print(iris_species)
print(iris.data)

# Store the inputs as a Pandas Dataframe and set the column names
x = pd.DataFrame(iris.data, columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width'])
 
y = pd.DataFrame(iris.target, columns = ['Targets'])

# Set the size of the plot
plt.figure(figsize=(14,7))
 
# Plot Sepal
plt.subplot(1, 2, 1) # Creating subplots (1st subplot of 1 row, 2 columns)

# Produce a scatter plot for the sepal length and width 
plt.scatter(x.Sepal_Length, x.Sepal_Width)
plt.xlabel('Length')
plt.ylabel('Width')
plt.title('Sepal')
plt.show()
plt.subplot(1, 2, 2)
# Produce a scatter plot for the petal length and width 
plt.scatter(x.Petal_Length, x.Petal_Width)
plt.xlabel('Length')
plt.ylabel('Width')
plt.title('Petal')
plt.show()

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Create scaler: scaler
scaler = StandardScaler()

# Create a PCA instance: pca
pca = PCA()

# Fit_transform scaler to 'X'
X_norm = scaler.fit_transform(x)

# Fit pca to 'X'
pca.fit(X_norm)
pca_features = pca.transform(X_norm)
# Plot the explained variances
features = range(0, pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()

from sklearn.cluster import KMeans

# Create a KMeans instance with 3 clusters: model
model = KMeans(n_clusters=3)

# Fit model to points
model.fit(X_norm)

# Determine the cluster labels of new_points: labels
labels = model.labels_
print(labels)

# Create a KMeans model with 3 clusters: model
model = KMeans(n_clusters=3)

# Use fit_predict to fit model and obtain cluster labels: labels
labels = model.fit_predict(pca_features)

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'species': iris_species})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['species'])

# Display ct
print(ct)
#k-means attempts to minimize the inertia when choosing clusters
print(model.inertia_)

ks = range(1, 6)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(pca_features)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()
