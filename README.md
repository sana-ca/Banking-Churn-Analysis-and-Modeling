
# 📊 Banking Customer Churn Analysis & Modeling

![Bank Image](https://m.economictimes.com/thumb/msid-100281493,width-1200,height-900,resizemode-4,imgsize-14062/banks-request-rbi-for-more-time-for-new-loan-provisioning-system.jpg)

This project provides an in-depth analysis of banking customer churn. The primary goal is to understand the factors that lead customers to leave the bank and to build a predictive machine learning model that can identify customers at risk of churning.

By predicting churn probability, banks can proactively engage with at-risk customers, minimize financial losses, and improve overall customer satisfaction.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Project Objective

1.  **Analyze Customer Churn:** Investigate the dataset to understand why customers leave the bank.
2.  **Identify Key Factors:** Use data analysis and machine learning to find the key drivers of customer attrition.
3.  **Build a Predictive Model:** Train and evaluate multiple classification models to predict the probability of a customer churning.
4.  **Provide Actionable Recommendations:** Based on the model's findings, suggest strategies for the bank to reduce churn.

---

## 📋 Table of Contents

- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)
- [Project Workflow](#-project-workflow)
- [Key Findings from EDA](#-key-findings-from-eda)
- [Model Performance](#-model-performance)
- [Key Churn Drivers (Feature Importance)](#-key-churn-drivers-feature-importance)
- [Business Recommendations](#-business-recommendations)
- [How to Run this Project](#-how-to-run-this-project)
- [License](#-license)

---

## 💾 Dataset

The project uses the **Churn Modelling** dataset, which contains 10,000 records of bank customers with 14 different attributes.

-   **Source:** [Kaggle - Churn Modelling Dataset](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)

---

## 🛠️ Tech Stack

-   **Data Manipulation & Analysis:** Pandas, NumPy
-   **Data Visualization:** Matplotlib, Seaborn
-   **Machine Learning:** Scikit-learn
-   **Imbalanced Data Handling:** Imblearn (SMOTE)
-   **Model Interpretation:** SHAP (SHapley Additive exPlanations)

---

## 🔄 Project Workflow

The analysis and modeling process followed these steps:

1.  **Data Wrangling:**
    -   Loaded the dataset and performed an initial assessment.
    -   Dropped irrelevant columns (`RowNumber`, `CustomerId`, `Surname`).
    -   Checked for and confirmed no missing values or duplicate records.

2.  **Exploratory Data Analysis (EDA):**
    -   Visualized the distribution of the target variable (`Churned`) to identify class imbalance.
    -   Analyzed the relationship between churn and categorical features like `Gender`, `Geography`, `HasCrCard`, and `IsActiveMember`.
    -   Analyzed the relationship between churn and numerical features like `CreditScore`, `Age`, `Tenure`, `Balance`, and `EstimatedSalary`.

3.  **Feature Engineering:**
    -   Created a `Total_Products` feature by grouping `NumOfProducts` into categories ('One product', 'Two Products', 'More Than 2 Products').
    -   Created an `Account_Balance` feature by categorizing `Balance` into 'Zero Balance' and 'More Than zero Balance'.

4.  **Data Preprocessing:**
    -   Applied **One-Hot Encoding** to categorical features.
    -   Performed **Log Transformation** on the `Age` feature to handle its right-skewed distribution.
    -   Used **SMOTE (Synthetic Minority Over-sampling Technique)** on the training data to address the class imbalance, ensuring the model isn't biased towards the majority class.

5.  **Model Training & Evaluation:**
    -   Trained two classification models: **Decision Tree** and **Random Forest**.
    -   Used **GridSearchCV** to perform hyperparameter tuning and find the optimal parameters for each model.
    -   Evaluated models based on Accuracy, Precision, Recall, F1-Score, and the ROC-AUC score.
    -   Used **SHAP** to interpret the model's predictions and understand the impact of each feature.

---

## 📊 Key Findings from EDA

-   **Geography:** Customers from **Germany** have a significantly higher churn rate compared to those from France and Spain, even though France has the highest number of customers.
-   **Gender:** **Female** customers have a higher probability of churning than male customers.
-   **Number of Products:** Customers with **1 product** or **more than 2 products** are much more likely to churn. Customers with **2 products** have the lowest churn rate.
-   **Active Status:** Customers who are **not active members** are almost twice as likely to churn.
-   **Age:** Older customers, particularly those in the **40-60 age range**, show a higher tendency to churn.

---

## 🚀 Model Performance

Both models performed well, but the **Random Forest Classifier** showed slightly better and more stable results.

| Metric              | Decision Tree | **Random Forest** |
| ------------------- | :-----------: | :---------------: |
| **Training Accuracy** |    86.84%     |     **87.52%**    |
| **Testing Accuracy**  |    80.20%     |     **86.40%**    |
| **F1-Score**        |     0.80      |      **0.86**     |
| **AUC Score**       |     0.84      |      **0.86**     |

The Random Forest model is the recommended model due to its higher accuracy on unseen data and better AUC score, indicating superior discriminative power.

---

## ⭐ Key Churn Drivers (Feature Importance)

The models identified the following features as the most significant predictors of churn:

1.  **Age:** The most influential factor. Older customers are more likely to churn.
2.  **NumOfProducts:** A critical factor. Customers with 1 or >2 products are at high risk.
3.  **Balance:** Customers with a non-zero balance are more likely to churn (often counter-intuitive, but may indicate they are moving their funds elsewhere).
4.  **IsActiveMember:** Inactive members are a major churn risk.
5.  **Geography (Germany):** Being a customer in Germany is a strong predictor of churn.

*This SHAP summary plot from the notebook visually confirms the impact of these features on the model's predictions.*
![SHAP Summary Plot](path/to/your/shap_plot.png)  <!-- **Action Required:** You need to save the SHAP plot from your notebook and add it to your repo, then update this path. -->

---

## 💡 Business Recommendations

Based on the analysis, the bank can take the following proactive steps to reduce customer churn:

1.  **Target Customers with One or Multiple (>2) Products:**
    -   **Action:** Develop marketing campaigns to encourage customers with a single product to acquire a second one (e.g., a credit card or a loan). This group has the lowest churn rate.
    -   **Action:** Investigate why customers with 3 or 4 products are churning. It may be due to poor integration or management of multiple services.

2.  **Engage with Senior and Middle-Aged Customers:**
    -   **Action:** Introduce special loyalty programs, benefits, or financial advisory services tailored for customers aged 40 and above.

3.  **Re-engage Inactive Members:**
    -   **Action:** Launch re-engagement campaigns via email, notifications, or calls, offering them incentives to use the bank's services. Highlight new features or benefits.

4.  **Investigate Issues in Germany:**
    -   **Action:** Conduct a region-specific analysis for Germany. This could involve reviewing local competition, service quality, or specific banking products offered in that region to understand the high churn rate.

---

## 🚀 How to Run this Project

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    A `requirements.txt` file is recommended. You can create one with:
    ```sh
    pip freeze > requirements.txt
    ```
    Then, install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```
    Alternatively, install them manually:
    ```sh
    pip install numpy pandas matplotlib seaborn scikit-learn imblearn shap jupyterlab
    ```

4.  **Download the dataset:**
    The notebook includes a cell to download the data directly from Kaggle using `kagglehub`. Ensure you have your Kaggle API token set up.
    ```python
    import kagglehub
    kagglehub.dataset_download('shrutimechlearn/churn-modelling')
    ```

5.  **Launch Jupyter and run the notebook:**
    ```sh
    jupyter lab
    ```
    Open the `📊_Banking_Churn_Analysis_&_Modeling.ipynb` file and run the cells.

---

