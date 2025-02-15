![IK](https://github.com/user-attachments/assets/f7830da1-1a7a-47b1-849f-f1bcae4aa441)
# 💡 IronKaggle Sales Prediction Challenge

Welcome to the Ironkaggle Sales Prediction project! This repo tackles a data science challenge to predict retail sales using historical data and key factors like customer traffic, promotions, holidays, and store operations.

This project follows a step-by-step machine learning workflow and demonstrates various techniques, from data preprocessing and feature engineering to training and evaluating multiple models for the best prediction accuracy.

---

## 📚 Table of Contents 

1. [Problem Statement](#problem-statement)
2. [Dataset Overview](#dataset-overview)
3. [Project Structure](#project-structure)
4. [Approach](#approach)
    - [Step 1: Data Exploration](#step-1-data-exploration)
    - [Step 2: Data Preprocessing](#step-2-data-preprocessing)
    - [Step 3: Feature Engineering](#step-3-feature-engineering)
    - [Step 4: Model Selection & Training](#step-4-model-selection-training)
    - [Step 5: Model Evaluation](#step-5-model-evaluation)
5. [Key Insights](#key-insights)
6. [Visualizations](#visualizations)
7. [Conclusion](#conclusion)
8. [Getting Started](#getting-started)
9. [Contributions](#contributions)
10. [License](#license)

---

## 🎯 Problem Statement 

The goal of this challenge was to predict the sales for a retail store based on multiple factors such as customer traffic, promotions, holidays, and whether the store is open.

---

### 📑 Dataset Overview 

The dataset contains sales data for a retail store, with the following columns:

| Column               | Description                                  |
|----------------------|----------------------------------------------|
| `True_index`          | Unique identifier for each observation      |
| `Store_ID`            | ID of the store                              |
| `Day_of_week`         | Day of the week (1-7)                        |
| `Date`                | Date of the observation                      |
| `Nb_customers_on_day` | Number of customers visiting the store       |
| `Open`                | Whether the store was open (1) or closed (0) |
| `Promotion`           | Whether a promotion was applied (1 or 0)     |
| `State_holiday`       | Type of state holiday (e.g., 'a', 'b', 'c')  |
| `School_holiday`      | Whether it was a school holiday (1 or 0)     |
| `Sales`               | Sales amount (target variable)               |

- **Total Rows**: 640,840
- **No Missing Values**: Clean dataset without missing or duplicated values
- **Shape**: (640840, 10)

---

## 📂 Project Structure  

This project is structured as follows:

- **Data Preprocessing**: The raw dataset was cleaned by removing unnecessary columns, and converting categorical data into numerical features.
- **Feature Engineering**: Additional time-related features such as year, month, day, and is_weekend were created to capture seasonal and weekly patterns that may influence sales.
- **Modeling**: Multiple regression models were tested and evaluated to predict sales. This included:
  - Linear Regression
  - Random Forest Regressor
  - AdaBoost Regressor
  - Gradient Boosting Regressor
  - XGBoost Regressor (Best performing model)
  - LightGBM Regressor

- **Model Evaluation**: The models were evaluated based on performance metrics like Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² score to identify the best model.

---

## 🛠️ Approach 

### 🔍 Step 1: Data Exploration 
- Explored the data to understand its structure, identified categorical features and numerical variables, and checked for any missing values or outliers.
- Visualized the correlation between features and the target variable (Sales) to understand which variables might have the most influence on sales predictions.

### 🧹 Step 2: Data Preprocessing 
- Dropped columns that weren't relevant for model prediction (such as `True_index`, and `open`, since selling while closed makes no sense), but ended up putting them back to avoid future problems.
- Ensured that missing and duplicated values were checked, although the dataset did not contain any.
- Renamed and ordered columns to common standards.

### 🧑‍💻 Step 3: Feature Engineering & Scaling 
- Added new time-based features such as `year`, `month`, `day`, and `is_weekend` to capture temporal patterns.
- Converted categorical features like `State_holiday` and `School_holiday` into numerical values using One-Hot Encoding.
- Since most models used are tree models that dont require scaling, only Standardization (Z-score scaling) has been done for Linear Regression and Normalization for KNN Regressor. 
  
### 🤖 Step 4: Model Selection & Training 
- Split the data into training and test sets (80% training, 20% test).
- Trained several regression models and tuned their hyperparameters to get the best performance.
- Evaluated each model using metrics like MAE, RMSE, and R² score.

### 📊 Step 5: Model Evaluation 
The performance of different models was compared to find the one with the best prediction power:

| Regressor             | Train Accuracy | Test Accuracy | Accuracy Difference |
|-----------------------|----------------|---------------|---------------------|
| Linear Regression     | 0.8481         | 0.8481        | 0.0000              |
| Random Forest         | 0.8175         | 0.8151        | 0.0023              |
| AdaBoost              | 0.7913         | 0.7932        | -0.0019             |
| Gradient Boosting     | 0.8949         | 0.8935        | 0.0014              |
| XGBoost               | 0.9495         | 0.9654        | -0.0159             |
| LightGBM              | 0.9116         | 0.9101        | 0.0015              |

The **XGBoost Regressor** showed the best results, with the highest test accuracy.

---

## 🔑 Key Insights 

- **Sales Drivers**: Sales were most strongly influenced by the number of customers visiting the store, whether a promotion was running, and certain holidays.
- **Temporal Patterns**: Weekend sales were typically lower than weekday sales, and state or school holidays sales showed noticeable spikes.
- **Promotion Impact**: Promotions significantly boosted sales.

---

## 📉 Visualizations 

Here are some of the key visualizations that helped understand the dataset and model results:

1. **Sales Distribution**: Distribution of sales across different days of the week.

![sales_distribution_across_weekdays](https://github.com/user-attachments/assets/54d759fb-8746-4bbd-922e-e1270013c882)
   
2. **Feature Correlation Heatmap**: A heatmap to visualize the correlation between features and the target variable.

![Correlation](https://github.com/user-attachments/assets/54f639d0-db4d-41b9-8dc4-a692d8f7ad3b)

3. **Model Performance Chart**: A bar chart showing the accuracy of each regression model.

![Regressors Comparison](https://github.com/user-attachments/assets/beca0ba5-3962-4d40-b0e6-d1713e8961d5)

4. **Best Model (XGB) Feature Importance**.

![Best_XGB_features_importance](https://github.com/user-attachments/assets/6f7c9161-1f81-41ac-8377-adfab842cee0)

---

## 🎯 Conclusion 

- **Best Model**: XGBoost performed the best out of all tested models. By tuning hyperparameters such as `n_estimators`, `max_depth`, and `learning_rate`, XGBoost was able to achieve the best performance in predicting sales.

  - Test MAE 554.4303498621464
  - Test RMSE 875.9190485201827
  - Test R2 score 0.9482756853103638
  - Train R2 score 0.967122495174408

---

## 🚀 Getting Started  

To replicate the work and run the models locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/Airam-DA/sales-prediction-ml-project.git

2.	Install the required libraries:
  	```bash
    pip install -r requirements.txt      

4. Run the Jupyter notebook or Python scripts to explore, preprocess, train models, and evaluate their performance.

--- 

# 🤝 Contributions

Contributions are welcome! Feel free to submit issues or pull requests to improve the project.

---

# 📝 License
This project is licensed under the **MIT License**, allowing you to share, modify, and use the code for both personal and professional purposes.



