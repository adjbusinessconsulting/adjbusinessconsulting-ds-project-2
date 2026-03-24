# DS Project 2 — Predicting Food Delivery Time with ML

## Problem Statement
If delivery time estimates are inaccurate, customers become disappointed, restaurants receive complaints, and operations teams struggle to allocate couriers. A ML prediction model allows delivery platforms to provide more realistic ETAs.

## What We Did
End-to-end ML pipeline: EDA on 1,000 delivery orders, feature engineering (Is_Rush_Hour flag, Distance × Prep_Time interaction term), preprocessing with StandardScaler + OneHotEncoder, then compared 3 regression models. The best model was selected based on MAE, RMSE, and R² score.

## Tools Used
- Python (Pandas, Scikit-Learn, Matplotlib, Seaborn)
- Jupyter Notebook / Google Colab

## Dataset
- **Source:** Food Delivery Times Dataset
- **Size:** 1,000 orders × 9 features

## Features
| Feature | Type | Description |
|---------|------|-------------|
| Distance_km | Numerical | Delivery distance |
| Weather | Categorical | Weather condition |
| Traffic_Level | Categorical | Traffic level |
| Time_of_Day | Categorical | Time of day |
| Vehicle_Type | Categorical | Courier vehicle type |
| Preparation_Time_min | Numerical | Kitchen preparation time |
| Courier_Experience_yrs | Numerical | Courier experience in years |

## Model Comparison
| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| **Linear Regression** | **6.08** | **8.96** | **0.8208** ✅ |
| Random Forest | ~6.5 | ~9.2 | ~0.81 |
| Gradient Boosting | ~6.8 | ~9.5 | ~0.80 |

> **Best Model: Linear Regression** with R² = 0.82 and MAE of only ±6 minutes.

## So What? — Actionable Insights

**1. Use predictive ETA models in the app**
Show customers more realistic delivery estimates. The model predicts with R²=0.82 and MAE of only ±6 minutes.

**2. Allocate couriers more strategically**
Assign experienced couriers to difficult routes or busy periods. Distance and courier experience are both significant factors.

**3. Adjust ETA during peak hours**
Incorporate rush hour and traffic conditions into operational planning. ETAs should automatically adjust for these factors.

**4. Monitor preparation time together with distance**
Delivery delay is not only caused by courier travel but also kitchen readiness. Distance × Prep_Time was one of the strongest predictors.

**5. Use the model as a decision-support tool**
The model can help operations teams anticipate delays and improve service quality proactively.

## Author
**Anthony Djiady Djie** — DS39+ Dibimbing.id
[ADJ Business Consulting](https://adjbusinessconsulting.github.io/adj-consulting)
