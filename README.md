<h1 align="center"> E-commerce customers clustering</h1>

 align="center">
  <img src="https://github.com/igorvgp/DS-clustering-ecommerce-insiders/blob/main/img/ecommerce.png" alt="drawing" width="800"/>
</p>

*The in-depth Python code explanation is available in [this](https://nbviewer.org/github/igorvgp/DS-clustering-ecommerce-insiders/blob/main/notebooks/insiders_clustering.ipynb#overview) Jupyter Notebook.*

# 1. **Abstract**
 This Data Science project was developed with data available on [Kaggle](https://www.kaggle.com/code/cheekonglim/uk-high-value-customers-identification) in order to find out the most valuable Ecommerce customers to join a loyalty program called "Insiders".
<p align="justify"> Gaussian Mixture unsupervised machine learning model was used for clustering into a Tree-Based embedding, where the "Insiders" cluster represents 15.7% of the customers and those customers are responsible for 51.7% of the company's Gross Revenue.</p>

<p align="justify"> The architecture of the project is shown in the image below: </p>
{}
<p align="center">
  <img src="https://github.com/igorvgp/DS-clustering-ecommerce-insiders/blob/main/img/diagram.jpg" alt="drawing" width="800"/>
</p>

The solution was deployed using Amazon Web Services (AWS) resources and the cluster analysis dashboard can be seen by clicking [here](https://datastudio.google.com/reporting/eadfbbd6-cc94-4d26-bd73-c642687f60fb/page/uj6AD?s=m6A3oybLvPw).

<p align="center">
  <img src="https://github.com/igorvgp/DS-clustering-ecommerce-insiders/blob/main/img/Insiders_Cluster_Analysis1.jpg" alt="drawing" width="350"/>
</p>

# 2. **Data Overview**
The data was collected from [Kaggle](https://www.kaggle.com/). This [dataset](https://www.kaggle.com/code/cheekonglim/uk-high-value-customers-identification) contains all the transactions occurring between Nov-2016 to Dec-2017 for a UK-based online retail store. The initial features descriptions are available below:

| Feature | Definition |
|---|---|
| InvoiceNo | A 6-digit integral number uniquely assigned to each transaction.|
| StockCode | Product (item) code. |
| Description | Product (item) name.|
| Quantity | The quantities of each product (item) per transaction.|
| InvoiceDate | The day when each transaction was generated.|
| UnitPrice | Unit price (Product price per unit).|
| CustomerID | Customer number (Unique ID assigned to each customer).|
| Country | Country name (The name of the country where each customer resides).|

# 3. **Assumptions**
- The Customer ID missing values were filled out with artificial customer IDs per purchase. 
- Some stock codes were removed as they do not represent completed purchases and behave like noise.
- 'European Community' and 'Unspecified' countries were deleted from the dataset as it doesn't represents a significant amout of data.
- Unit price below 4 cents have been removed from the data, since it's not the original price of the products and they interfere with modeling the phenomenon.
- The user "16446" was removed from the dataset because he has a very disparate number of purchases and returns. likely related to fraud or error.
- The dataset was grouped by users and new features were created in order to best describe the problem: 

| New Feature | Definition |
|---|---|
| Gross Revenue | Unit price * quantity of products.|
| Recency | Days since last purchase.|
| qtty_invoices| Quantity of invoices by user.|
| qtty_items | Quantity of items purchased by user.|
| qtty_products | Quantity of different products purchased by user.|
| avg_ticket | The average ticket for each user: sum(gross_revenue) / no_invoices|
| avg_recency_days | The average shopping interval time.|
| frequency | Quantity of purchased products per day.|
| qtty_returns | Quantity of returned products.|
| basket_size | Average quantity of products per purchase.|
| unique_basket_size | Average quantity of unique products per purchase.|

# 4. **Solution Plan**
## 4.1. How was the problem solved?

<p align="justify"> To select the most valuable costumers to join a loyalty program called "Insiders" a Unsupervised Machine Learning model was applied. To achieve that, the following steps were performed: </p>

<b> Understanding the Business Problem </b> :

* Who are the people eligible for the loyalty program?
* What does "being elegible" mean? 
* Who are the "valuable customers"?
* Invoicing:
	* High ATP
	* High LTV
	* Low recency
	* High basket size
	* Low probability of churn
* Cost:
	* Low product return rate
* Shopping Experience:
	* High average rating

- <b> Collecting Data </b>: Collecting e-commerce data from Kaggle.

- <b> Data Cleaning </b>: Renaming columns, changing data types and filling NaN's. 

- <b> Feature Engineering </b>: Creating new features from the original ones, so that those could be used in the ML model. 

- <b> Data Preparation </b>: Applying Rescaling Techniques in the data.

- <b> Space Studying </b>: Analysing the variability of the data after applying PCA, UMAP, t-SNE and tree based embeddind.

- <p align="justify"> <b> Machine Learning Modeling </b>: Applying clustering algorithms with different K values to find the one with best performance. </p>

- <b> Model Evaluation </b>: Evaluating the model using Silhouette Score and Silhouette Diagram. 

- <p align="justify"> <b> Exploratory Data Analysis (EDA) </b>: Exploring the data in order to analyze the characteristics of the clusters and look for useful business insights.

- <p align="justify"> <b> Model Deployment </b>: Model deployment through aws resources (S3, EC2 and RDS). </p>

-<p align="justify"><b> Model Monitoring</b>: Dashboard developed with Google Looker Studio to visualize the clusters and its main matrics. </p>It can be accessed by clicking [here](https://datastudio.google.com/reporting/eadfbbd6-cc94-4d26-bd73-c642687f60fb/page/uj6AD?s=m6A3oybLvPw).
  
## 4.2. Tools and techniques used:
- Python 3.9.13
- Pandas and Numpy
-  Matplotlib, Seaborn and Sklearn.
- Jupyter Notebook.
- AWS Services: S3: raw data, EC2: virtual machine (Ubuntu 20.04), RDS: Database (PostgreSQL).
- Embedding methods: PCA, t-SNE, UMAP and Tree Based embedding.
- Clustering algorithms (K-Means, Hierachical Clustering, DBSCAN and HDBSCAN).
- Performance Metrics: Silhouette Score and Silhouette Diagram.

# 5. **Machine Learning Models**

<p align="justify"> A tree-based embedding with two features ensured greater separation of possible clusters, so four clustering ML algorithms were tested on this space space: </p>

- K-Means
- Hierarchical Clustering
- DBSCAN
- HDBSCAN

The performance was calculated through [Silhouette Score](https://towardsdatascience.com/silhouette-coefficient-validating-clustering-techniques-e976bb81d10c). The initial performance for all four algorithms are displayed below: 

<p align="center">
  <img src="https://github.com/igorvgp/DS-clustering-ecommerce-insiders/blob/main/img/algorithms_silhouette.png" alt="drawing" width="800"/>
</p>


<p align="justify"> KMeans, Gaussian Mixture Model and Hierarchical Clustering presented the best Silhouettes, reaching the score of 0.71. Despite that, analyzing the graphical distribution of the clusters it is possible to observe attributions and it was possible to identify inconsistent attributions.</p>

<p align="justify"> <b> The Gaussian Mixture Model with 10 clusters was chosen, presenting a good Silhouette score of 0.68 and a good cluster distribution</b>. </p>

<p>After applying Gaussian mixture model to the data, the following cluster distribution was obtained: </p>

<p align="center">
  <img src="https://github.com/igorvgp/DS-clustering-ecommerce-insiders/blob/main/img/gmm_clusters.png" alt="drawing" width="800"/>
</p>

## 5.1. Summary of results:

<p align="justify"> Below is a table with the main characteristics of each group of customers found, the clusters were named according to these characteristics. </p>

<p align="center">
  <img src="https://github.com/igorvgp/DS-clustering-ecommerce-insiders/blob/main/img/clusters_summary.png" alt="drawing" width="800"/>
</p>

# 6. **Model Deployment**

As previously mentioned, the results are available on a Dashboard available [here](https://datastudio.google.com/reporting/eadfbbd6-cc94-4d26-bd73-c642687f60fb/page/uj6AD?s=m6A3oybLvPw). 

<p align="justify">  The idea behind this is to facilitate access from anywhere and any mobile device with internet access. With the information available on the Dashboard, the marketing team will be able to analyze the behavior of customer groups and create personalized action plans for each of them.

In addition, the data scientist will be able to monitor the model by identifying the shape of the clusters with the entry of new clients in the Tree Based Embedding space, identifying the union or generation of new clusters in the space, and evaluating the retraining results.

</p>

# 7. **Conclusion**
<b>In this project the main objective was accomplished:</b>

 <p align="justify">The most valuable customers were found: A group of 15.8% of the customers that brings 51.7% of the company's revenue, that is called "Insiders". </p>
 <p align="justify">To reach this result, a Clustering algorithm called the Gaussian Mixture model was used to identify the groups in a Tree-Based embedding space.</p>
 <p align="justify">In addition to the insiders group, other groups of customers were found, with the aim of assisting in decision making. </p>

# 7. **Next Steps**

- Start a new CRISP cicle with the aim of testing the embedding spaces with more than two dimensions.
- Automate the retraining process.

 # Contact

- igorviniciusgpereira@gmail.com
- [![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/igorvgpereira/)
