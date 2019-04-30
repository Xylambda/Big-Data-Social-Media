import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.cluster import KMeans

# Pandas settings to expand display area
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


# ============= LOAD AND VISUALIZE DATA =============
# ---- Load iris dataset ----
iris = datasets.load_iris()

# ---- Transform iris dataset to dataframe ----
iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)

# ---- Print iris dataframe ----
print(iris_df)

# ---- Plot settings ----
plt.xlabel("petal length (cm)")
plt.ylabel("petal width (cm)")
plt.style.use('ggplot')

# ---- Visualize iris dataframe ----
plt.scatter(iris_df["petal length (cm)"], iris_df["petal width (cm)"], cmap = "viridis")

plt.show()


# ============ CLUSTER DATA USING KMEANS ============
# ---- Create a separated variable ----
X = [[iris_df["petal length (cm)"][i], iris_df["petal width (cm)"][i]] for i in range(0, len(iris_df))]

# ---- Algorithm settings ----
kmeans = KMeans(n_clusters = 3, random_state = 0)

# ---- Run the algorithm and predict clusters ----
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# ---- Visualize clustered data ----
plt.scatter(iris_df["petal length (cm)"], iris_df["petal width (cm)"], c = y_kmeans, s = 50, cmap = 'viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c = 'black', s = 100, alpha = 0.5)

plt.show()


# ===================== CREDITS =====================
### Jake VanderPlas:
# https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html
    
### Aletta Smits:
# Big Data and Social Media / Data Learning Class (week 1 - Day 1)
