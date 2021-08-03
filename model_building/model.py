import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn import metrics
from yellowbrick.cluster import KElbowVisualizer

# directories
BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
DATA_DIR = os.path.join(BASE_DIR, 'feature_eng', 'data', 'ft_df.csv')
# MODEL_DIR = os.path.join(BASE_DIR, 'heroku', 'models')

df = pd.read_csv(DATA_DIR)

df.drop(columns = 'customer_id', inplace = True)

# function to test metrics in k clusters
def clustering_algorithm(n_clusters, dataset):
    values = dataset.values
    values = MinMaxScaler().fit_transform(values)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(values)
    s = metrics.silhouette_score(values, labels, metric='euclidean')
    dbs = metrics.davies_bouldin_score(values, labels)
    calinski = metrics.calinski_harabasz_score(values, labels)
    return print(s, dbs, calinski)

for i in range(2,10):
    print(f'{i} cluster(s)')
    clustering_algorithm(i, df)

# Instantiate the clustering model and visualizer
model = KMeans(n_init=10, max_iter=300)
visualizer = KElbowVisualizer(model, k=(4,12))

# scaling dataset
values = df.values
values = MinMaxScaler().fit_transform(values)

visualizer.fit(values)        # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure


# algorithm
kmeans = KMeans(n_clusters = 7, n_init = 10, max_iter = 300)
y_pred = kmeans.fit_predict(values)
labels = kmeans.labels_

df['cluster'] = labels


# summary
description = df.groupby("cluster")
n_clients = description.size()
description = description.mean()
description['n_clients'] = n_clients
description['n_clients_perc'] = n_clients/n_clients.sum()*100 

description.round(2)
