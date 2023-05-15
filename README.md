# Introduction

In this project, I will be predicting the churn for customers at a bank. The dataset is for ABC Multistate Bank and was obtained from Kaggle (https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset). The dataset includes 12 columns and 10,000 rows, where each row represents a customer. The columns it includes are:
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

I will start by performing exploratory data analysis to attempt to find any patterns that may be apparent in the data and will make various plots and tables to do this. This will allow me to identify what features might be the biggest drivers of churn. Next, I will perform data preprocessing to clean the data, perform feature engineering, feature selection, data transformation, and normalizing or standardizing. Finally, I will select an appropriate machine learning algorithm and train and evaluate the model on how well it is able to predict the churn. Once the model is trained and evaluated, I will attempt to optimize the model to improve its accuracy at predicting customer churn. 

## Hypotheses

I am going to assume that `age` may be a big predictor in churn, where older customers are more likely to churn. I think this because younger people do not really put too much thought into their bank because in many cases, that is taken care of by the parents, so I am going to assume that many younger people just follow the bank that their parents go to and do not consider switching banks very often. 

I believe `balance` also could be a big driver in churn because customers with a very low balance (especially a balance of $0) are more likely to churn because it is much easier for a person to switch banks if they do not have to worry about transferring money from one bank to another. This could be the same case for `tenure`, where people that have been a customer at the bank for a shorter period of time might be more likely to churn.

Another feature I think would be a big driver in churn is `active_member`. I think it is fairly obvious to assume people that are not a current active member will have a higher churn rate than people that are an active member, because they have already left the bank, so this might show some abnormal findings in the data. 

# EDA: Descriptive Statistics and Visualizations

**INCLUDE SUMMARY STATS TABLE HERE**

Our binary categorical variables are `credit_card`, `active_member`, and `churn`. From this summary, we can see that about 70% of the customers in the dataset have a credit card (based on the mean `credit_card`), about 50% of the customers in the dataset are an active member, and we have many more customers in the dataset that did not churn, compared to those who did churn (roughly 20% of the customers churned). Also, the `customer_id` column will be irrelevant for our model, so we can drop that column when we perform the data cleaning. 

We see that we have an imbalance target variable, `churn`. It has a mean of 0.2037, meaning that 20.37% of the customers have churned, while almost 80% have not. This imbalance could lead to a biased model that may not perform well on the minority class (churned customers). We should fix this imbalance by oversampling the minority class, undersampling the majority class, or using a combination of both (e.g., SMOTE)

We also see that some features, such as `credit_score`, `age`, `balance`, and `estimated_salary` have different ranges and scales. This might affect the performance of certain ML algorithms, such as k-Nearest Neighbors or SVM's, which are sensitive to the scale of input features. Therefore, we should standardize or normalize these numerical features before using them.

We also notice that `country` and `gender` are not included in this summary because they are categorical variables, therefore we need to encode them before using our ML algorithms, especially ones that require numerical input. 

## Correlation heatmap

![Plot 1: Heatmap](images/plot1-heatmap.png)

We see that `age` has the highest positive correlation with churned customers. Based on this and our exploratory graphs, we can say that older customers are more likely to churn than younger ones. 

Since the heatmap only includes our numerical variables, lets take a look at churn rate by gender and country to see if there are any interesting patterns.

## Country and gender

![Plot 2: Country and Gender](images/plot2-country-gender.png)

To do this, we calculated the churn rate and plotted our results. 

We find that females have a significantly higher churn rate than males in all three countries. We also find that Germany has the highest churn rate among the three countries for both genders. Lastly, we find that France and Spain have very similar churn rates, with Spain having a slightly higher churn rate for both genders. 

However, we should note that the number of customers in each country is different, with France having the most and Spain having the least. A smaller sample size can cause biased results in churn rate, so this should be considered when interpreting these results. We are also unsure if the difference in churn rates between countries is statistically significant, which should also be taken into consideration when interpreting these results. If we wanted to further examine that, we could conduct a hypothesis test to answer that question. 

**TLDR: Germany churns the most. Females churn more than males.**

## Age groups

To explore how age and churn are related, I created age groups and calculated the churn rates for each age group. The following are the age groups I created and the counts of how many customers the bank has in those age groups:

| **Age Group** | **Count** |
|---------------|-----------|
| 18-30         |      1946 |
| 31-45         |      5921 |
| 46-60         |      1647 |
| 60+           |       464 |

The age group with the most customers is the 31-45 age group by far. The age group with the least amount of customers is the 60+ age group, which has the least number of customers by far. Now, let's take a look at their churn rates.

![Plot 3: Age Groups](images/plot3-age-groups.png)

The age group with the highest churn rate is the 46-60 age group by far. This age group actually churns more than they do not. The age group that churns the least is the 18-30 age group. So, we can see that older people tend to churn more often than younger people. What is causing this to happen?

**TLDR: Older people churn more than younger people. More specifically, ages 46-60.**

## Balance and credit score

![Plot 4: Balance and Credit Score](images/plot4-balance-creditscore.png)

## Number of products 

![Plot 5: Products in Countries](images/plot5-products.png)

Wow! 100% of customers that have 4 products from the bank all churned. This might indicate a genuine issue with customer retention when customers have 4 products. Let's see how many customers have 4 products in the dataset to see why this might be happening. Is it because we only have a couple customers with 4 products and they just so happened to leave? Or is there enough customers that have 4 products from the bank where we should start investigating why this is happening?

Looking at the number of people in each country that have 1, 2, 3, or 4 products, I find the following counts:

| **_products_number_** | **France** | **Germany** | **Spain** |
|-----------------------|------------|-------------|-----------|
| **1**                 |       2514 |        1349 |      1221 |
| **2**                 |       2367 |        1040 |      1183 |
| **3**                 |        104 |          96 |        66 |
| **4**                 |         29 |          24 |         7 |

We find that the majority of customers have 1 or 2 products, the churn rate is relatively lower for customers with 1 or 2 products than those with 3 or 4 products, and the number of customers with 3 or 4 products is much lower compared to those with 1 or 2 products. This means the high churn rates for customers with 3 or 4 products is less representative of the general population due to the small sample size. However, the very high churn rates still warrants further investigation because 60 customers (customers with 4 products) is not a small enough number to where this can be brushed off or happened due to chance. Let's see a visualization of these results.

## Important Observations

1. Around 20% of customers have churned.
2. Germany has the customers that churn the most by far. Customers in France and Spain churn about equally, with French people churning slightly less. 
2. Females churned more than males.
3. Older people churn more than younger people, especially in the 46-60 age group. The 46-60 age group is the only group that churns more than they do not. Age is also the variable with the highest correlation with churned customers. 
4. Customers with more products from the bank churn more than customers with less products. We found that customers with 4 products churned 100% of the time. Do more credit card influence spending habits in a negative way, which is causing this?

# Data Preprocessing

Before getting into our predictions, I needed to perform the data cleaning, feature selection, and data preprocessing. 

The first thing I did was drop the columns that were not going to be relevant for our machine learning model. The columns I dropped were `customer_id` and `age_bins` (since I added this column when I created the age groups for the EDA).

Next, I needed to encode the categorical variables in the dataset. To do this, I used LabelEncoder on the `country` and `gender` columns of the dataset. After doing the encoding, France = 0, Germany = 1, Spain = 2 in the `country` feature and for the `gender` variable, Female = 0, Male = 1.

## Feature Selection

The next step was to perform feature selection. Since we have a mix of numerical and categorical input variables, a popular approach to feature selection is to use tree-based methods, such as decision trees or Random Forests because they can handle both numerical and categorical input variables and provide feature importances that can be used for feature selection. You can use the feature importances provided by a Random Forest model, for example, to rank the input variables and then select the top K most important features. This is the exact approach I used for feature selection.

The plot below shows the rankings of the feature importances that were provided by the Random Forest model.

![Plot 6: Feature Importances](images/plot6-feature-importances.png)

As we can see in the plot, there is a pretty significant drop after feature 5 (`products_number`), so we will only take the top 5 features. Therefore, we find that the 5 most important features are `age`, `estimated_salary`, `credit_score`, `balance`, `products_number`.

Once this was done, I then created my training and testing sets. Now, I am ready to select a model and make some predictions. 

# Model Selection and Evaluation

Before selecting any models, I wanted to calculate the null error rate. The null error rate is the measure of how wrong the classifier would be if it just predicted the majority class, which in this case is the 0 in our `churn` column. In other words, predicting when a customer will not churn. I found the null error rate to be 0.8035, or roughly 80%, which means that a dumb model that always predicts 0 would be correct about 80% of the time. This tells us the minimum we should achieve with our model. 

After training and evaluating a logistic regression model, a neural network classifier, an optimized neural network classifier (optimized with hyperparameter tuning), and an XGBoost classifier model, I found the best model to be the XGBoost classifier. Here is a peak at the classification report of that model:

|   | **Precision** | **Recall** | **F1-Score** |
|---|---------------|------------|--------------|
| 0 |          0.80 | 0.76       | 0.78         |
| 1 |          0.78 | 0.81       | 0.79         |

I also found the accuracy to be 0.788 and the AUC to also be 0.788.

This model is quite good compared to the neural network model and logistic regression model. The accuracy shows the model is correct about 79% of the time. The precision, recall, and F1-score for both classes are quite balanced, which suggests the model performs equally well for both classes. The AUC score is also considered acceptable, almost considered excellent (0.8-0.9). 

# Model Optimization

To see if I could improve this model's performance, I wanted to handle the class imbalance that was present in the data and also find the best parameters to use. To address the class imbalance, I oversample the minority class using SMOTE. Then, I performed hyperparameter tuning and searched for the best parameters using GridSearchCV. Once the best parameters were found, I was ready to rerun the model and evaluate it. Below is a peak at the classification report.

|   | **Precision** | **Recall** | **F1-Score** |
|---|---------------|------------|--------------|
| 0 |          0.81 | 0.77       | 0.79         |
| 1 |          0.78 | 0.82       | 0.80         |

I also found the accuracy to be 0.796 and the AUC to also be 0.796.

So, tuning the hyperparameters gave us a very, very slight improvement in the model's performance. To visualize both of the models performances, below is a graph showing the ROC curves.

![Plot 7: ROC Curves](images/plot7-roc-curves.png)
