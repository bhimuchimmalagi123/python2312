import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
iris=load_iris()
x=iris.data
y=iris.target
print("original dataset shape:{x.shape}")
scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)
pca=PCA(n_components=2)
x_pca=pca.fit_transform(x_scaled)
print("pca after shape:{x_pca.shape}")
print("explained variance ratio",pca.explained_variance_ratio_)
plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=y,cmap='viridis')
plt.xlabel('principal component 1')
plt.ylabel('principal component 2')
plt.title('pca of iris dataset')
plt.colorbar()
plt.show()
