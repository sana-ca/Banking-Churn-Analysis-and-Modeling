📊 Banking Churn Analysis & Modeling
This project analyzes customer churn behavior in a banking dataset using data visualization and exploratory data analysis (EDA) techniques in Python. The goal is to uncover insights about which customer attributes are associated with churn, using various statistical plots and group-level analyses.

🔍 Objective
To perform detailed churn analysis on a bank’s customer dataset and visualize how features such as tenure, balance, age, and credit score impact churn behavior.

📁 Dataset
The dataset contains customer information, including:

CreditScore

Geography

Gender

Age

Tenure

Balance

NumOfProducts

HasCrCard

IsActiveMember

EstimatedSalary

Churned (Target variable)

📊 Key Analyses
Countplots of churn grouped by categorical features like Gender, Geography, Tenure.

Histograms & KDE plots to show how continuous variables (e.g., Age, Balance) differ for churned vs. non-churned customers.

Boxplots to detect outliers and compare distributions across churn groups.

Bar plots with percentage annotations to better understand proportions in each group.

📌 Example Visualizations
Customer Churn by Tenure with percentage labels on each bar.

Age Distribution by Churn Status using histogram and boxplot side by side.

Balance vs. Churned comparison to understand account activity patterns.

📦 Tech Stack
Python 3.x

Pandas

Seaborn

Matplotlib

Jupyter Notebook

📈 Status
✔️ Data cleaned
✔️ EDA complete
❌ Predictive modeling (e.g., LSTM) not implemented in this version

🔮 Future Enhancements
Add churn prediction using Machine Learning models (e.g., Logistic Regression, Random Forest, LSTM).

Create an interactive dashboard (Plotly or Streamlit).

Perform feature engineering and correlation analysis.

