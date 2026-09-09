# 🚲 Bike Rental Prediction & Analysis

## 📌 Overview
This repository contains a complete workflow for analyzing bike rental data and building predictive models. It combines exploratory data analysis, visualization, and machine learning to understand rental patterns and forecast demand.

## 📂 Repository Structure
- **BikeRent_Prediction_model.ipynb**  
  Jupyter Notebook with the end-to-end predictive modeling pipeline. Includes data preprocessing, feature engineering, model training, and evaluation.

- **Bike_Rental_Data_Analysis_Report.docx**  
  A detailed report summarizing insights from the dataset, including trends by day, hour, and external factors.

- **Rental_by_day.csv**  
  Dataset containing daily rental counts with relevant features (e.g., weather, season, holidays).

- **Rental_by_hour.csv**  
  Dataset with hourly rental counts, useful for fine-grained demand analysis.

- **README.md**  
  Documentation for the repository (this file).

## 🎯 Objectives
- Perform **data analysis** to uncover rental trends and seasonal patterns.
- Build a **prediction model** to forecast bike rental demand.
- Provide actionable insights for **resource allocation** and **business planning**.

## 🛠️ Tools & Technologies
- **Python** (data analysis & modeling)
- **Jupyter Notebook** (interactive workflow)
- **Pandas, NumPy** (data manipulation)
- **Matplotlib, Seaborn** (visualization)
- **Scikit-learn** (machine learning models)
- 
## 📈 Workflow Diagram
![Bike Rental Prediction Workflow](https://copilot.microsoft.com/th/id/BCO.6af7cb58-454c-46a2-b00e-9f4a69656038.png)
### 📈 Comparison Matrix

| Rank | Model | MAE (↓) | RMSE (↓) | R² Score (↑) | MAPE (%) (↓) | Evaluation Summary |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| 🥇 **1st** | **CatBoost** | **0.14** | **0.24** | **0.75** | **1.79%** | **Clear Winner:** Dominates all four metrics with the highest fit and lowest overall error. |
| 🥈 **2nd** | **XGBoost** | 0.16 | 0.26 | 0.70 | 2.00% | **Strong Runner-Up:** Shares $R^2$ and RMSE with Gradient Boosting but delivers lower MAE and MAPE. |
| 🥉 **3rd** | **Gradient Boosting** | 0.17 | 0.26 | 0.70 | 2.11% | **Solid Baseline:** Reliable tree-based performance; ties XGBoost on explained variance ($R^2 = 0.70$). |
| 4th | **LightGBM** | 0.18 | 0.27 | 0.67 | 2.21% | **Moderate:** Demonstrates improved performance over SARIMAX but trails the other tree models. |
| 5th | **SARIMAX** | 0.27 | 0.33 | 0.51 | 3.41% | **Baseline:** Classical linear time-series framework; struggles with non-linear feature interactions. |

---

### 🔍 Key Insights from model

* **CatBoost Dominance:** CatBoost decisively outperforms all models across every single evaluation metric:
  * Highest explained variance ($R^2 = 0.75$).
  * Lowest absolute deviation ($\text{MAE} = 0.14$).
  * Best resistance to large residual outliers ($\text{RMSE} = 0.24$).
  * Tightest relative percentage error ($\text{MAPE} = 1.79\%$).
* **XGBoost vs. Gradient Boosting:** While both achieve an identical $R^2$ of **0.70** and RMSE of **0.26**, XGBoost gains an advantage with a lower MAE (**0.16** vs. **0.17**) and lower MAPE (**2.00%** vs. **2.11%**).
* **Ensemble Tree Models Outperform Statistical Approaches:** All gradient-boosted tree algorithms ($R^2$ between **0.67** and **0.75**) significantly outshine SARIMAX ($R^2 = 0.51$), proving that non-linear feature interactions are essential to explaining target variance in this dataset.

---

### 🏆 Final Recommendation

**CatBoost Regressor** is the recommended final model for deployment due to its distinct performance lead and superior handling of categorical and non-linear interactions.

## 📊 Key Insights
- Rental demand varies significantly by **time of day** and **season**.
- External factors such as **weather conditions** and **holidays** influence usage.
- Predictive models can help optimize **inventory management** and **service availability**.

## 📢Challenges Faced

### Challenge 1: Data Leakage from casual and registered
**The problem:** casual + registered = cnt exactly, for every single row. Including these columns as model inputs would let the model "cheat" by learning the trivial identity cnt = casual + registered, achieving near-perfect accuracy that would completely fall apart in production, since on a real, not-yet-happened day, you don't know how many casual vs. registered riders will show up.

**How I handled it:** I verified this relationship explicitly with an assert statement before dropping both columns from the feature set, so the decision to exclude them was backed by evidence, not assumption.

### Challenge 2: Multicollinearity Between temp and atemp
**The problem:** temp and atemp were correlated at r=0.99, with very high VIF scores (>40-50), meaning they carry almost identical information. Feeding both into a linear model creates unstable, unreliable coefficients.

**How I handled it:** I dropped atemp and kept temp, verified through the correlation heatmap and VIF table, preserving the temperature signal while removing the redundancy.

### Challenge 3: Small Dataset Size (Only 731 Rows)
**The problem:** With just 584 training rows, tree-based ensemble models (Random Forest, Decision Tree) are prone to overfitting, and any single train/test split carries real risk of not being representative.

**How I handled it:** I used GridSearchCV with cross-validation to tune hyperparameters (like max_depth and min_samples_leaf) specifically to constrain model complexity, and confirmed that Decision Tree and Random Forest, despite tuning, still underperformed boosting models, which is a genuine limitation of this dataset's size rather than a tuning failure.

### Challenge 4: Chronological vs. Random Train/Test Splitting
**The problem:** This is time-series data. A standard random 80/20 split would let the model train on some future dates and test on earlier ones, creating unrealistic "lookahead" leakage that inflates performance in a way that wouldn't hold up in real deployment.

**How I handled it:** I used a strict chronological split, the first 584 days for training, the last 147 days for testing, ensuring the model is only ever evaluated on dates it hasn't seen and that come after its training window, just like a real production scenario.

### Challenge 5: The Hurricane Sandy Outlier (October 29, 2012)
**The problem:** On this single day, rentals collapsed to just 22 (from a typical 5,000+), due to Hurricane Sandy shutting down the bike-share system. None of the available weather columns (weathersit, temp, windspeed) captured the severity of this event, weathersit=3 on that day just means "light rain," identical to any ordinary rainy Tuesday.

**How I handled it:** I kept this real data point in the test set (rather than removing it, which would be manipulating results) but switched from raw MAPE, which broke down near this near-zero value, spiking to over 12,000% error on this single day to more robust metrics: **WAPE and SMAPE**, which stay stable and meaningful even with this kind of extreme outlier present.

