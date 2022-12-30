<h1 align="center"> E-commerce costumers clustering</h1>

<p align="center">
  <img src="https://user-images.githubusercontent.com/66283452/195202224-01bfd468-9f1c-4e83-af60-b101312a98e3.svg" alt="drawing" width="800"/>
</p>

*The in-depth Python code explanation is available in [this](https://nbviewer.org/github/igorvgp/DS-clustering-ecommerce-insiders/blob/main/notebooks/insiders_clustering.ipynb#overview) Jupyter Notebook.*

# 1. **Abstract**
 This Data Science project was developed with data available on [Kaggle](https://www.kaggle.com/code/cheekonglim/uk-high-value-customers-identification) in order to find out the most valuable Ecommerce customers to join a loyalty program called "Insiders".
<p align="justify"> Gaussian Mixture unsupervised machine learning model was used for clustering into a Tree-Based embedding, where the "Insiders" cluster represents 15.7% of the customers and those customers are responsible for 51.7% of the company's Gross Revenue.</p>

<p align="justify"> The architecture of the project is shown in the image below: </p>

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
| qtty_products | Quantity of different products purchased by user.|
| qtty_products | Quantity of different products purchased by user.|
| qtty_products | Quantity of different products purchased by user.|

# 4. **Solution Plan**
## 4.1. How was the problem solved?

<p align="justify"> To predict sales values for each store (six weeks in advance) a Machine Learning model was applied. To achieve that, the following steps were performed: </p>

- <b> Understanding the Business Problem </b> : Understanding the reasons why Rossmann's CEO was requiring that task, and plan the solution. 

- <b> Collecting Data </b>: Collecting Rossmann store and sales data from Kaggle.

- <b> Data Cleaning </b>: Renaming columns, changing data types and filling NaN's. 

- <b> Feature Engineering </b>: Creating new features from the original ones, so that those could be used in the ML model. 

- <p align="justify"> <b> Exploratory Data Analysis (EDA) </b>: Exploring the data in order to obtain business experience, look for useful business insights and find important features for the ML model. </a>. </p>

- <b> Data Preparation </b>: Applying Normalization and Rescaling Techniques in the data, as well as Enconding Methods and Response Variable Transformation.

- <b> Feature Selection </b>: Selecting the best features to use in the ML model by applying the <a href="https://www.section.io/engineering-education/getting-started-with-boruta-algorithm/">Boruta Algorithm</a>. 

- <p align="justify"> <b> Machine Learning Modeling </b>: Training Regression Algorithms with time series cross-validation. The best model was selected to be improved via Hyperparameter Tuning. </p>

- <b> Model Evaluation </b>: Evaluating the model using four metrics: MAE, MAPE and RMSE. 

- <b> Financial Results </b>: Translating the ML model's statistical performance to financial and business performance.

- <p align="justify"> <b> Model Deployment (Telegram Bot) </b>: Implementation of a Telegram Bot that will give you the prediction of any given available store number. This is the project's <b>Data Science Product</b>, and it can be accessed from anywhere. </p>
  
## 4.2. Tools and techniques used:
- Python 3.9.13, Pandas, Matplotlib, Seaborn and Sklearn.
- Jupyter Notebook and VSCode.
- Flask and Python API's.  
- Ngrok and Telegram Bot.
- Git and Github.
- Exploratory Data Analysis (EDA). 
- Techniques for Feature Selection.
- Regression Algorithms (Linear and Lasso Regression; Random Forest and XGBoost Regressors).
- Cross-Validation Methods, Hyperparameter Optimization and Algorithms Performance Metrics (RMSE, MAE and MAPE).

# 5. **Machine Learning Models**

<p align="justify"> This was the most fundamental part of this project, since it's in ML modeling where the sales predictions for each store can be made.  An average model was used as a baseline and four models were trained using time series cross-validation: </p>

- Linear Regression
- Lasso Regression (Regularized Linear Regression)
- Random Forest Regressor
- XGBoost Regressor

<p>The baseline model performance is displayed below: </p>

<div align="center">

| **Model Name** | **MAE** | **MAPE** | **RMSE** |
|:---:|:---:|:---:|:---:|
| Average Model | 1354.80 | 0.2064	 | 1835.135542 |

</div>

<p>The initial performance for all four algorithms are displayed below: </p>

<div align="center">

| **Model Name** | **MAE** | **MAPE** | **RMSE** |
|:---:|:---:|:---:|:---:|
| Random Forest Regressor | 1104.87 +/- 209.75 | 0.16 +/- 0.03	 | 1530.38 +/- 273.38 |
| XGBoost Regressor | 1179.33 +/- 111.96 | 0.17 +/- 0.01	 | 1639.3 +/- 148.34 |
| Linear Regression | 2079.0 +/- 280.91 | 0.3 +/- 0.01	 | 2955.05 +/- 426.56 |
| Lasso Regression | 2090.34 +/- 307.94 | 0.3 +/- 0.01 | 2995.12 +/- 458.89 |

</div>

<p align="justify"> Both Linear Regression and Lasso Regression have worst performances in comparison to the simple Average Model. This shows a nonlinear behavior in our dataset, hence the use of more complex models, such as Random Forest and XGBoost. </p>

<p align="justify"> <b> The XGBoost model was chosen for Hyperparameter Tuning. Even if Random Forest has the best performance if we look into the metrics, XGBoost would still be better to use, because it's much faster to train and tune </b>. </p>

<p>After tuning XGBoost's hyperparameters using Random Search the model performance has improved: </p>

<div align="center">
	
| **Model Name** | **MAE** | **MAPE** | **RMSE** |
|:---:|:---:|:---:|:---:|
| XGBoost Regressor | 949.881428	 | 0.143602 | 1336.919406 |


</div>

## 5.1. Brief Financial Results:

<p align="justify"> Below there are displayed two tables with brief financial results given by the XGBoost model. </p>

<p align="justify"> A couple interesting metrics to evaluate the financial performance of this solution is the MAE and MAPE. Below there's a table with a few stores metrics: </p>
<div align="center">

| **Store** | **Predictions (€)** | **Worst Scenario (€)** | **Best Scenario (€)** | **MAE (€)** | **MAPE** |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1  |	164,545.94 |	150,086.63 |	179,005.24 |	14,459.31 |	0.09 |
| 2  |	178,759.59 |	151,883.56 |	205,635.62 |	26,876.03 |	0.15 |
| 3  |	266,517.19 |	231,827.11 |	301,207.26 |	34,690.07 |	0.13 |
| 4  |	340,026.47 |	303,667.24 |	376,385.70 |	36,359.22 |	0.10 |
| 5  |	170,492.62 |	132,908.07 |	208,077.14 |	37,584.53 |	0.22 |
</div>

<p align="justify"> According to this model, the sales sum for all stores over the next six weeks is: </p>

<div align="center">

| **Scenario (€)** | **Total Sales of the Next 6 Weeks (€)** |
|:---:|:---:|
| Prediction  | $283,742,272.00 |
| Worst Scenario | $244,033,471.48 |
| Best Scenario | $323,451,121.16 |

</div>

# 6. **Model Deployment**

<p align="justify">  As previously mentioned, the complete financial results can be consulted by using the Telegram Bot. The idea behind this is to facilitate the access of any store sales prediction, as those can be checked from anywhere and from any electronic device, as long as internet connection is available.  
The bot will return you a sales prediction over the next six weeks for any available store, <b> all you have to do is send him the store number in this format "/store_number" (e.g. /12, /23, /41, etc) </b>. If a store number if non existent the message "Store not available" will be returned, and if you provide a text that isn't a number the bot will ask you to enter a valid store id. 

To link to chat with the Rossmann Bot is [![image](https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white)](https://t.me/rossman_newapi_bot)

<i> Because the deployment was made in a free cloud (Render) it could take a few minutes for the bot to respond, <b> in the first request. </b> In the following requests it should respond instantly. </i>

</p>

# 7. **Conclusion**
In this project the main objective was accomplished:

 <p align="justify"> <b> A model that can provide good sales predictions for each store over the next six weeks was successfully trained and deployed in a Telegram Bot, which fulfilled CEO' s requirement, for now it's possible to determine the best resource allocation for each store renovation. </b></p>

 # Contact

- igorviniciusgpereira@gmail.com
- [![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/igorvgpereira/)
