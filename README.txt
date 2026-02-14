
MobAI Forecasting Models - Production Package
==============================================

FILES:
------
1. xgboost_regression_model.json - Regressor (predicts demand amount)
2. xgboost_classifier_model.json - Classifier (predicts P(demand > 0))
3. forecast_config.json - Model configuration
4. product_reference.csv - Product metadata

USAGE:
------
1. Load both XGBoost models
2. Load config to get feature names and threshold
3. For each product:
   - Create feature vector (22 features)
   - Get probability from classifier
   - If probability > 0.4: use regressor prediction
   - Else: predict 0

FEATURES (22 total):
-------------------
Lags: demand_lag_1, 7, 14, 30
Rolling: rolling_mean_7, 14, 30, rolling_std_7
Calendar: day_of_week, week_of_year, month, is_weekend
Intermittency: days_since_demand, demand_freq_30d, demand_cv, rolling_max_30
Product: categorie_encoded, Poids(kg), volume pcs (m3), cat_mean_demand, cat_std_demand, cat_total_demand

PERFORMANCE:
-----------
WAPE: 111.36% (baseline: 155.42%)
Bias: 0.82% ✅ (target: 0-5%)
Improvement: 28.4 percentage points

DATE: February 13, 2026
