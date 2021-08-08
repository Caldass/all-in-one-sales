import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn import metrics
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
import umap.umap_ as umap

# directories
BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
DATA_DIR = os.path.join(BASE_DIR, 'feature_eng', 'data', 'ft_df.csv')

# MODEL_DIR = os.path.join(BASE_DIR, 'heroku', 'models')

df = pd.read_csv(DATA_DIR)
df.drop(columns = ['customer_id', 'mix'], inplace = True)



## Hyperparameter fine tuning

# scaling dataset
values = df.values
values = MinMaxScaler().fit_transform(values)

# function to test metrics in k clusters
def clustering_algorithm(n_clusters, dataset):
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(values)
    s = metrics.silhouette_score(values, labels, metric='euclidean')
    dbs = metrics.davies_bouldin_score(values, labels)
    calinski = metrics.calinski_harabasz_score(values, labels)
    return print(s, dbs, calinski)

for i in range(2,10):
    print(f'{i} cluster(s)')
    clustering_algorithm(i, values)

# elbow method visualizer
model = KMeans(n_init=10, max_iter=300)
visualizer = KElbowVisualizer(model, k=(4,12))

visualizer.fit(values)        # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure



## Model Training

# algorithm
kmeans = KMeans(n_clusters = 3, n_init = 10, max_iter = 300)
y_pred = kmeans.fit_predict(values)
labels = kmeans.labels_

df['cluster'] = labels



## Visualization Inspection

# silhouette visualizer
visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick')

visualizer.fit(values)        # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure

# Using UMAP
df_viz = df.copy()
reducer = umap.UMAP( n_neighbors=90, random_state=42 )
embedding = reducer.fit_transform( values )

# embedding
df_viz['embedding_x'] = embedding[:, 0]
df_viz['embedding_y'] = embedding[:, 1]

# plot UMAP
sns.scatterplot( x='embedding_x', y='embedding_y', 
                 hue='cluster',
                palette=sns.color_palette( 'hls', n_colors=len( df_viz['cluster'].unique() ) ),
                 data=df_viz , legend= list(df_viz.cluster.unique()))
plt.show()



## Cluster Profile

description = df.groupby("cluster")
n_clients = description.size()
description = description.mean()
description['n_clients'] = n_clients
description['n_clients_perc'] = n_clients/n_clients.sum()*100 

description.round(2)
