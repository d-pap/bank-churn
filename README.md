# Introduction

In this project, I will be predicting the churn for customers at ABC Multinational Bank. I will start by performing exploratory data analysis to attempt to find any patterns and identify what features might be the biggest drivers of churn. Next, I will perform data preprocessing to prepare the data for modeling. I will then test out a few different models and select the best one to use for predicting churn. Finally, I will generate a list of customers who are at risk of churning. These customers can then be targeted with marketing campaigns to try to retain them.

## Dataset Information

The dataset was obtained from Kaggle (https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset) and includes 12 columns and 10,000 rows, where each row represents a customer. The columns it includes are:
* `customer_id` (numerical, a unique ID given to identify each customer)
* `credit_score` (numerical, a score that identifies the creditworthiness of a customer)
* `country` (categorical, represents the country where the customer is located - it can be one of three countries: Spain, Germany, or France), `gender` (categorical, represents the customer's gender)
* `age` (numerical, represents the customer's age)
* `tenure` (numerical, representing how long the customer has been at the bank given in years)
* `balance` (numerical, representing the amount of USD the customer has at the bank)
* `products_number` (numerical, representing the number of products the customer has at the bank)
* `credit_card` (is a binary categorical variable, representing if the customer has a credit card at the bank or not - 0 = no, 1 = yes)
* `active_member` (is a binary categorical variable, representing if the customer is an active member at the bank)
* `estimated_salary` (numerical, representing the estimated total income of the customer in USD)
* `churn` (used as the target variable, is a binary categorical variable and represents if the customer has left the bank during some period of time - 0 = they have not, 1 = they have). 

**Please see the .ipynb file to see the full report.**


