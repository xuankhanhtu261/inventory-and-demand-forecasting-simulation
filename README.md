# Retail Demand Forecasting and Inventory Risk Management Using Time Series Modeling and Monte Carlo Simulation

A data-driven framework combining time series forecasting, Monte Carlo simulation, and inventory policy evaluation to help retailers balance stockout risk against excess inventory under demand uncertainty.

## Overview

Point forecasts alone don't tell you how much safety stock to hold — real demand deviates from any single prediction. This project builds a full pipeline that goes from raw sales data to a concrete, risk-aware inventory policy recommendation:

1. **Forecast** future demand using four models of increasing complexity
2. **Simulate** thousands of plausible demand futures via Monte Carlo (bootstrap-resampled forecast errors)
3. **Evaluate** inventory replenishment policies under that simulated uncertainty
4. **Recommend** a policy configuration that balances stockout risk against inventory holding cost

Applied to two years of daily sales data for a single grocery product (Store S001, Product P0005) from the [Retail Store Inventory and Demand Forecasting dataset](https://www.kaggle.com) (Kaggle, synthetic).

## Key Findings

- **SARIMAX** was the best-performing forecasting model (MAE 25.47, RMSE 32.17, MAPE 27.15%), outperforming Seasonal Naive, Holt-Winters ETS, and XGBoost — driven by its ability to incorporate exogenous factors (Promotion, Epidemic, Weather, Seasonality) alongside temporal dependence.
- **Epidemic conditions and Promotions** were the strongest demand drivers identified by both SARIMAX coefficients and XGBoost feature importance. Promotion is the only one of these directly controllable by the business.
- **Promotions increase demand but also increase stockout risk** (13.3% → 18.0% under a fixed reorder policy) unless replenishment strategy is adjusted accordingly.
- **A reorder-point (s, S) policy with safety factor k = 2.33** achieved the best tradeoff: ~1.7–3.4% stockout probability while holding only ~245–260 units of average ending inventory — roughly 60% less inventory than a fixed periodic policy needed to reach a comparable service level.

## Methodology

### 1. Exploratory Data Analysis
- Time series decomposition (trend / seasonality / residual), stationarity testing (ADF), rolling statistics
- ACF/PACF analysis for temporal dependence structure
- Correlation and multicollinearity checks (Discount excluded due to 0.80 correlation and structural confounding with Promotion)

### 2. Forecasting Models
| Model | Role |
|---|---|
| Seasonal Naive | Baseline benchmark |
| Holt-Winters ETS | Captures trend/seasonality from demand history alone |
| **SARIMAX** (winner) | Adds autoregressive/moving-average structure + exogenous regressors |
| XGBoost | ML approach with engineered lag/calendar/exogenous features |

SARIMAX order `(1,0,2)(0,0,0)[7]` selected via `auto_arima` (AIC-optimized).

### 3. Monte Carlo Demand Simulation
- SARIMAX out-of-sample residuals bootstrap-resampled (non-parametric — residuals showed heavy tails per Q-Q plot) and added to point forecasts
- 1,000 simulated 60-day demand paths per scenario
- Two scenarios: **No Promotion** vs. **Weekly Promotion**

### 4. Inventory Policy Simulation
- **Policy A** — fixed periodic reorder (every 5 days, no safety margin)
- **Policy B** — reorder-point (s, S) policy, `s = demand×lead_time + k×σ_resid`
- Evaluated via stockout probability, daily stockout rate, and average ending inventory across all 1,000 simulated paths
- Sensitivity analysis on buffer size (Policy A) and safety factor *k* (Policy B)

## Repository Structure

```
├── data/
│   └── retail_store_inventory.csv       # raw dataset (not included — see Data source)
├── notebooks/
│   └── inventory_time_series_forecasting.ipynb   # full analysis pipeline
├── report/
│   └── project_report.pdf               # full written report
└── README.md
```

## Requirements

```
pandas
numpy
matplotlib
seaborn
statsmodels
pmdarima
xgboost
scikit-learn
```

## Data Source

[Retail Store Inventory and Demand Forecasting]([https://www.kaggle.com](https://www.kaggle.com/datasets/atomicd/retail-store-inventory-and-demand-forecasting/data)) — synthetic dataset, Kaggle. Filtered to Store ID `S001`, Product ID `P0005` (Groceries), 760 daily observations (2022-01-01 to 2024-01-30).

## Limitations

- Synthetic dataset; single product/store — findings may not generalize
- Lead time (1 day) assumed, not empirically identifiable from the data
- Residuals resampled independently across days (no temporal dependence in simulated errors)
- Single train/test split rather than rolling-origin cross-validation
- Simplified inventory model (no ordering costs, storage constraints, or product expiry)

See the full report for a complete discussion of limitations and future work.

