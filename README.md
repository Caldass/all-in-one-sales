# All in One Place Customer Clustering - Project Overview
- Identified customers in an e-commerce database to be part of a loyalty program.
- Created 10+ features in order to segment the customer database.
- Used UMAP + Tree based embedding + KMeans algorithm in order to cluster the customers.
- Automated the process of predicting new customers clusters using Amazon Web Services RDS, EC2 and Storage services.

## Code and Resources Used
**Python version:** 3.8.12 <br />
**Packages:** Pandas, Pandas Profilling, Numpy, UMAP, Sklearn, Pickle, Seaborn, Matplotlib, Yellowbrick, Scipy, SQLAlchemy, s3fs. <br />
**Data:** https://github.com/provezano/UK-High-Value-Customers-Identification/blob/main/data/Ecommerce.csv

## The company All In One Place
The company All in One Place is a Multibrand Outlet company, that is, it sells second-line products of several brands at a lower price, through an e-commerce.

In just over 1 year of operation, the marketing team realized that some customers in its base buy more expensive products with high frequency and end up contributing with a significant portion of the company's revenue.

Based on this perception, the marketing team will launch a loyalty program for the best customers in the base, called Insiders. But the team does not have an advanced knowledge of data analysis to elect program participants.

For this reason, the marketing team asked the data team to select eligible customers for the program, using advanced data manipulation techniques.

##  Data Description
- Filled missing customer ids.
- Fixed data types.
- Explored the dataframe's descriptive statistics.

## Data Filtering
- Filtered out stock codes that contained strings.
- Filtered out 0 prices.
- Separated revenue and returns into two different dataframes.
- Removed countries not identified.
- Removed discrepant users identified in the EDA step.

## Feature Engineering
Transformed the dataframe into a unique customer dataframe, creating the following features regarding each customers full data:
- **avg_recency_days**: The average amount of days it took for a customer to buy again.
- **avg_basket_size**: The average size of the customer's basket.
- **avg_unique_basket_size**: The average size of the unique products of a customer's basket.
- **gross_revenue**: The customer's gross revenue.
- **qt_returns**: The amount of returns a customer made.
- **last_purchase**: The amount of days since the last purchase of a customer.
- **orders**: The amount of orders of a customer.
- **qt_products**: The unique amount of products a customer bought.
- **qt_items**: The total amount of products a customer bought.
- **frequency**: The frequency of purchase of a customer.
- **average_ticket**: The customer's average ticket. 

## EDA
In this step, I evaluated a pairplot of the features present in the data, along with an analysis of a couple of statistics regarding these features in order to identify which features had more variability, therefore, may segment better the data. Here are some of the highlights of the data exploration:
![alt text](https://github.com/Caldass/all-in-one-sales/blob/master/img/pairplot.png "Pairplot")
![alt text](https://github.com/Caldass/all-in-one-sales/blob/master/img/distributions.png "distributions")

After exploring the data, the following features were chosen to be used in the clustering algorithms:
- gross_revenue
- last_purchase
- avg_recency_days
- qt_products
- qt_returns

## Machine Learning Modeling
I transformed the data into a two dimensional space using a Tree Based Embedding using the Random Forest algorithm and the UMAP Transformer, since having an explainable model wasn't needed in this particular project:
![alt text](https://github.com/Caldass/all-in-one-sales/blob/master/img/tree_embedding.png "Embedding")

After that, I explored models such as KMeans, GMM (Gaussian Mixture Model), Hierarchical Clustering and DBSCAN, using the silhouette score metric to evaluate them. I chose this metric because it not only considers in its score the intra-cluster distance of the samples (distance within the cluster) but it also considers the distance between clusters. 

KMeans hat the best results considering 8 clusters alongside GMM, with a Silhouette Score of ~0.62. 

Given those results, I chose KMeans over GMM as the final model to cluster ou clients given that the data points formed during the embedding phase in my perspective would be better suited using a model that creates clusters in spherical shapes.

The DBSCAN model doesn't need any number of clusters to be set previously, differently from the other models I quoted. But still, its results were not as satisfying as the results provided by KMeans/GMM.

Here's a glance at how our customer database was segmented by the algorithm (the values are presented as means from each feature regarding the clusters):
![alt text](https://github.com/Caldass/all-in-one-sales/blob/master/img/clusters.jpg "clusters")


## Convert Model Performance to Business Values
As a result of this project, it was expected to submit a list of people eligible to participate in the Insiders program, along with a report answering the following questions:

1. Who are the people eligible to participate in the Insiders program?
- The people inside the cluster 4 (insiders).


2. How many customers will be part of the group?
- 468 people.


3. What are the main characteristics of these customers?
- Top revenue.
- Lowest buying lag.
- Lowest buying lag between purchases.
- Top amount of products purchased.
- Highest number of returns.
- Represent a big chunk of the client base.


4. What is the percentage of revenue contribution, coming from Insiders?
- 51.72%


5. What are the conditions for a person to be eligible for Insiders?
- The model created decides who's eligible.


6. What are the conditions for a person to be removed from Insiders?
- The model created decides who should be removed.


7. What is the guarantee that the Insiders program is better than the rest of the base?
- There's no guarantee at first, we could run an AB test to make shure that the insiders cluster performs better than the other clusters. Then, we could separate a A amount of clients from the insiders cluster and a B amount of clients from the other clusters for a specific amount of time and perform an AB testing.


8. What actions can the marketing team take to increase revenue?
- Thinking about the insiders group, which will be the focused group in the insiders project, one could suggest the marketing team to create a VIP type service for the insiders group, this way, avoiding those clients to end up in other clusters and loosing the characteristics that fit them in the insiders cluster. If we think about the other clusters, one could suggest the marketing team to take one of the clusters and try to analyze what they could do to boost those clients into better groups or even improve that group itself. For example using cluster number 2 which contains the biggest chunk of clients of the whole base, and trying to make those clients buy more products could result in a revenue increase. Maybe one could suggest the marketing team to create a sort of cross sell strategy with those clients.

### Top 3 Data Insights

#### H1: Customers in the insiders cluster represent more than 10% of the purchases volume (products).
True. The insiders cluster represent approximately 54% of the purchases volume (products).

#### H2: Insiders cluster customers have an average return amount below the average of the total return amount.
False. The insiders cluster has an avg return amount of 149.27, while the avg. total return amount is 34.89

#### H3: The median gross revenue of customers in the insider cluster is at least 10% higher than the median gross revenue of all customers.
True. The median gross revenue of the insiders cluster is only 276.48 % higher than all the median gross revenue.


## Deploying Model to Production
In this step, I automated the process of predicting new customers clusters using Amazon Web Services RDS, EC2 and Storage services. 

Through S3, I was able to store the initial data that would be consumed throughout the other steps performed in the cloud project.

Through RDS, I was able to create a MySQL database in order to keep historical and current data regarding the e-commerce's customers. 

Through EC2, I was able to create an Ubuntu server in which I used papermill and cronjob in order to periodically execute the transformation and modeling regarding the customers of the e-commerce, then, the output of the modeling would be inserted into the MySQL(RDS) database.

The output of the full process would be through a visualization tool such as Tableau or Metabase which would allow one to consume, present and analyze the data from the MySQL database.

Here's a glance at the full schema created in this process:

![alt text](https://github.com/Caldass/all-in-one-sales/blob/master/img/deploy.png "Deploy Structure")

