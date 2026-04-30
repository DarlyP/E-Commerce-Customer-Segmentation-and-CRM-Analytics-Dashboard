![E-Commerce-Customer-Segmentation Image](https://github.com/DarlyP/E-Commerce-Customer-Segmentation-and-CRM-Analytics-Dashboard/blob/main/wallpaper.jpg)

# E-Commerce Customer Segmentation and CRM Analytics Dashboard

---

## Tools

[<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />](https://www.python.org/)
[<img src="https://img.shields.io/badge/Jupyter-FA0F00?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter" />](https://jupyter.org/)
[<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" />](https://pandas.pydata.org/)
[<img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy" />](https://numpy.org/)
[<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Scikit-learn" />](https://scikit-learn.org/)
[<img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logoColor=white" alt="Matplotlib" />](https://matplotlib.org/)
[<img src="https://img.shields.io/badge/Power%20BI-F2C811?style=for-the-badge&logo=powerbi&logoColor=000000" alt="Power BI" />](https://powerbi.microsoft.com/)

---

## Dashboard & Dataset

**Dataset**:
[Shopify Dataset](https://www.kaggle.com/datasets/usernam3/shopify-app-store)

**Clean Dataset**:
[Clean Dataset](https://drive.google.com/drive/folders/1lDijxK1qgvfMAcm20mCm-G4KBvZ6Oi3M?usp=drive_link)

**Power BI Dashboard**:  
[Dashboard - E-Commerce-Customer-Segmentation-and-CRM-Analytics-Dashboard](https://drive.google.com/file/d/1PXJR5yyHzpKeSHHQbEwj6UBoF15P65aw/view?usp=drive_link)

**Presentation Deck / PDF**:  
[Presentation - E-Commerce-Customer-Segmentation-and-CRM-Analytics-Dashboard](https://docs.google.com/presentation/d/1k7iGdoQgGU5xfZH95QukVWq1iNvT0wZC/edit?usp=drive_link&ouid=110854372150850374714&rtpof=true&sd=true)

---

## Introduction

This project simulates an **end-to-end customer analytics workflow** for e-commerce data, with a focus on:

- **customer segmentation**
- **revenue quality analysis**
- **retention risk detection**
- **CRM action prioritization**
- **dashboard-ready outputs for business users**

The project goes beyond simple reporting by combining **value, behavior, engagement, loyalty, leakage signals, and retention risk** into one analytical framework.

The goal is not only to identify who spends more, but also to answer questions such as:

- Which customers create **healthy and stable value**?
- Which customers look valuable but are actually **fragile or risky**?
- Which customers should be **protected, developed, monitored, or recovered**?

---

## Business Context

In many businesses, customer analysis stops at basic metrics such as:

- monthly spend
- purchase frequency
- average order value

However, these metrics alone do not explain:

- whether revenue is **healthy or leaking**
- whether customers are **stable or starting to decline**
- which segments deserve **retention investment**
- which groups are too dependent on discounts
- which customers should be targeted for **upsell or CRM growth**

This project solves that by building a structured customer analytics framework that translates raw customer data into:

- segmentation insights
- revenue quality signals
- retention risk flags
- business action priorities

---

## Project Objectives

The main aims of this project are to:

- Segment customers based on **behavioral, value, and engagement** characteristics
- Analyze customer patterns related to **revenue quality** and **retention outcomes**
- Identify differences in:
  - spending
  - purchase frequency
  - loyalty
  - promotion dependency
  - risk behavior
- Detect early retention-risk signals such as:
  - weak engagement
  - cart abandonment
  - return behavior
  - unstable value contribution
- Compare **stable** and **at-risk** customers
- Generate business insights to support:
  - segmentation strategy
  - retention planning
  - customer prioritization
  - CRM actions

---

## Deliverables

This project produces:

- **FM segmentation** based on Frequency and Monetary behavior
- **Behavior-based customer segments**
- **Composite customer scores**, including:
  - Value Core
  - Behavior Core
  - Leakage Pressure
  - Loyalty Strength
- **Revenue quality analysis**
- **Retention risk scoring and categories**
- **Stable vs risky customer comparison**
- **Business priority mapping**
- **Dashboard-ready outputs** for Power BI

---

## Analytical Workflow

The project is organized as a notebook-driven analytics workflow:

### 1. Data Initialization
Prepares and validates the dataset for analysis.

### 2. Exploratory Data Analysis
Explores customer distributions, variable quality, and important behavioral patterns.

### 3. Shopper Behavior Analysis
Examines customer activity, spend behavior, and engagement-related metrics.

### 4. Behavioral Customer Segmentation and Retention Risk Analysis
Builds the main analytical logic:
- FM scoring
- behavior segmentation
- revenue quality analysis
- retention risk framework
- business priority mapping

### 5. Dashboard Preparation
Prepares dimension and fact outputs for BI/dashboarding.

---

## Statistical & Analytical Methods Used

This project combines **descriptive analytics**, **rule-based customer scoring**, and **light unsupervised learning** to build interpretable customer segments and retention-risk insights.

### 1. Descriptive Statistics
Basic descriptive statistics were used to understand the structure of the customer base, including:

- count
- mean
- median
- standard deviation
- minimum and maximum
- coefficient of variation

These statistics helped identify how customer value, engagement, and risk were distributed across the dataset.

---

### 2. Duplicate Validation
A duplicate assessment was performed on selected customer-level fields to ensure that only valid and non-duplicated customer records were used in the analysis.

This step was important to make sure that segmentation and scoring results reflected real customer patterns rather than repeated observations.

---

### 3. Winsorization (1%–99%)
Winsorization was applied to selected skewed numeric variables to reduce the effect of extreme outliers while keeping all customer records in the dataset.

This was especially important for variables such as:

- monthly spend
- weekly purchases
- lifetime value proxy
- risk-adjusted value
- return rate
- cart abandonment
- risk score

The goal was to make segmentation, scoring, and clustering more robust and more representative of the overall customer base.

---

### 4. Min-Max Normalization
Min-Max scaling was used to transform multiple variables into a common **0–1 range** before combining them into composite scores.

This ensured that variables with larger numeric ranges did not dominate the final score simply because of scale differences.

This method was used in the construction of:

- Value Core
- Behavior Core
- Leakage Pressure
- Loyalty Strength
- Retention Risk Score

---

### 5. Quantile-Based FM Scoring
Customer Frequency and Monetary behavior were scored using **quantile-based ranking**.

- **Frequency (F)** was based on weekly purchases
- **Monetary (M)** was based on monthly spend

Each variable was divided into **five equal-sized groups**, producing scores from **1 to 5**.

This approach was chosen because customer data is usually skewed, and quantile-based grouping creates more balanced and interpretable segments than fixed thresholds.

---

### 6. Weighted Composite Scoring
Several interpretable composite scores were created using a **weighted scoring model**.

The project used weighted combinations of normalized variables to summarize different dimensions of customer quality:

- **Value Core**
- **Behavior Core**
- **Leakage Pressure**
- **Loyalty Strength**
- **Retention Risk Score**

The weights were assigned using **business logic**, not black-box optimization, so the final scores remain explainable and practical for business interpretation.

---

### 7. Inverse Transformation for Risk Alignment
For variables that represent customer strength rather than weakness, such as:

- stability score
- retention strength

the normalized values were inverted using:

- `1 - normalized value`

This was done so that all components in the retention-risk framework followed the same direction:

> higher value = higher retention risk

This makes the final risk score easier to interpret.

---

### 8. Quantile-Based Risk Thresholding
After the retention risk score was created, customers were grouped into operational categories using quantile cutoffs.

The following thresholds were used:

- 50% → Stable
- 80% → Watchlist
- 95% → At Risk
- above 95% → Critical Risk

This method allows customers to be classified based on **relative risk position** in the dataset rather than fixed business cutoffs.

---

### 9. Group-Level Comparative Analysis
Segment profiling and risk comparison were performed using group-level aggregation and comparison across metrics such as:

- monthly spend
- weekly purchases
- engagement
- adjusted revenue
- return rate
- cart abandonment
- risk-adjusted value
- retention strength

This helped identify how stable customers differ from risky customers and how behavior segments differ in terms of value quality.

---

### 10. Revenue Quality Labeling
Revenue segments were classified into:

- **Healthy Revenue**
- **Revenue with Leakage Pressure**

This labeling was based on rule-based comparisons using:

- adjusted revenue
- return rate
- cart abandonment

The purpose was to distinguish strong top-line revenue from revenue that may be weakened by leakage behavior.

---

### 11. Rule-Based Business Prioritization
A final business-priority layer was created by combining:

- behavior segment
- retention risk flag

This produced action-oriented groups such as:

- Protect and Retain
- Urgent Retention Intervention
- Upsell and Value Expansion
- Retention Watchlist
- Discount Control and Margin Protection
- Low-Cost Automation
- Maintain and Monitor

This approach translates statistical analysis into practical CRM and retention actions.

---

### 12. KMeans Clustering Evaluation (Exploratory)
As an exploratory step, **KMeans clustering** was tested using multiple values of **k**.

Cluster quality was evaluated using **Silhouette Score** to assess how well-separated the clusters were.

The results showed that:
- **k = 2** gave the best score among the tested values
- but the overall silhouette values remained low

This suggests that natural cluster separation in the dataset is relatively weak, and that **business-rule segmentation is more interpretable and actionable** than unsupervised clustering alone.

---

## Why these methods were chosen

The project was designed to balance:

- **statistical robustness**
- **business interpretability**
- **actionable CRM output**

Rather than relying only on complex modeling, the analysis emphasizes methods that are:

- explainable
- reproducible
- suitable for dashboarding
- easy to translate into retention and segmentation decisions

## Repository Structure

```text
.
├── Notebook_IPYNB/
│   ├── Behavioral_Customer_Segmentation_and_Retention_Risk_Analysis.ipynb
│   ├── Dashboard_Preparation.ipynb
│   ├── Data_Initiate.ipynb
│   ├── Exploratory_Data_Analysis_of_Customer_Behavior_and_Profiles.ipynb
│   └── Shopper_Behavior_Analysis.ipynb
│
├── Notebook_Python/
│   ├── Behavioral_Customer_Segmentation_and_Retention_Risk_Analysis.py
│   ├── Dashboard_Preparation.py
│   ├── Data_Initiate.py
│   ├── Exploratory_Data_Analysis_of_Customer_Behavior_and_Profiles.py
│   └── Shopper_Behavior_Analysis.py
│
└── README.md

```

---

**Disclaimer**: 
- This notebook is created solely for learning and exploration purposes. There is no intention to offend or harm any party. All content and analysis presented are based on publicly available data online. I undertake this process to enhance my understanding of data analysis techniques and methodologies and hone my skills in implementing relevant algorithms and models within the context of data science learning. In conducting this analysis, I strive to maintain objectivity and professionalism in interpreting the existing data. Any conclusions or recommendations provided result from personal analysis and are not intended as professional advice in any specific capacity. I hope the information obtained from this notebook can be useful to anyone reading it to learn and develop data analysis skills.
