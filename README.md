# MobAI'26 - Warehouse Optimization & Demand Forecasting

## Team Solution Overview

This project implements an end-to-end warehouse management solution addressing:
- **Task 1**: Intelligent warehouse storage assignment & picking route optimization
- **Task 2**: Demand forecasting using Prophet per-SKU + XGBoost classifier with ensemble blending

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Requirements**: Python 3.10+ (tested on 3.13), ~2 GB RAM minimum.

### 2. Run Inference (Standalone Scripts)

#### Task 2 – Demand Forecasting

```bash
python inference_forecast.py \
  --input data/sample_test_forecast.csv \
  --output forecast_submission.csv \
  --start_date 2026-02-15 \
  --end_date 2026-03-17
```

**Input CSV** columns: `date, id_produit, quantite_demande` (historical demand)
**Output CSV** columns: `Date, id_produit, quantite_demande` (predictions, date format DD-MM-YYYY)

#### Task 1 – Warehouse Optimization

```bash
python inference_optimization.py \
  --input data/sample_test_optimization.csv \
  --output optimization_results.csv
```

**Input CSV** columns: `Date, Product, Flow Type, Quantity` (warehouse events)
**Output CSV** columns: `Product, Action, Location, Route, Reason`

### 3. Run the API (Optional)

```bash
python main.py
```

Access the API at `http://localhost:8000`. Interactive docs at `/docs`.

### 4. Run Tests

```bash
python main.py &           # Start server in background
python test_api.py          # Run 11 tests
```

### 5. Retrain Models (Optional)

```bash
python retrain_model.py
```

Requires the original Excel data file in `data/`.
Training time: ~15-20 minutes (Prophet per-SKU fitting + XGBoost training).

---

## Project Structure

```
mobai-api/
├── main.py                         # FastAPI v2.0 (all endpoints)
├── inference_forecast.py           # Task 2 standalone inference
├── inference_optimization.py       # Task 1 standalone inference
├── retrain_model.py                # Full training pipeline (v9)
├── training_notebook.ipynb         # Training notebook (documented)
├── test_api.py                     # API tests (11/11 passing)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── models/                         # Saved model artifacts
│   ├── xgboost_classifier_model.json   # Binary demand classifier
│   ├── xgboost_regression_model.json   # Quantity regressor
│   ├── xgboost_lf_regressor_model.json # LF segment regressor
│   ├── forecast_config.json            # Thresholds & feature list
│   ├── prophet_meta.json               # Per-SKU Prophet metadata
│   ├── product_attributes.json         # Physical product attributes
│   ├── delivery_stats.json             # Delivery pattern stats
│   └── cat_encoding.json               # Category encoding map
│
├── data/                           # Data files
│   ├── product_priorities.csv          # Product priority scores
│   ├── product_segments.csv            # HF/LF classification
│   ├── warehouse_locations.csv         # 837 warehouse locations
│   ├── sample_test_forecast.csv        # Sample forecast input
│   └── sample_test_optimization.csv    # Sample optimization input
│
└── __pycache__/
```

---

## Task 2: Demand Forecasting – Technical Approach

### Architecture: Prophet + XGBoost Classifier + Ensemble Blend

| Stage | Component | Description |
|-------|-----------|-------------|
| 0 | Prophet per-SKU | 571 products fitted (≥10 demand-days), unified params: `cps=0.01`, `sps=0.1`, multiplicative, 5 temporal regressors |
| 1 | XGBoost Classifier | Binary demand prediction (AUC=0.9375), 123 features, threshold=0.120 |
| 2 | Blend + Bias | `alpha=1.05 * prophet + (1-alpha) * ewm_7`, bias_mult=0.94 |

### Feature Engineering (123 Features)
- **Lag features**: 1, 2, 3, 7, 14, 21, 28 day lags
- **Rolling stats**: mean, std, max, min, median, sum over 3/7/14/28/60 day windows
- **Demand frequency**: Rolling demand occurrence rate (7/14/28/60 days)
- **Days since last demand**: Time gap feature
- **Calendar**: day-of-week, month, week, quarter, Fourier terms (yearly/weekly/monthly)
- **Prophet-derived**: yhat, trend, weekly/yearly components, residuals, ratios
- **Product static**: demand frequency, priority score, segment, physical attributes
- **HF×temporal interactions**: segment-specific calendar effects

### Performance

| Metric | Value |
|--------|-------|
| WAPE (demand-days) | 73.88% |
| Bias (demand-days) | 0.62% |
| Classifier AUC | 0.9375 |
| Improvement vs Lag-1 | 31.7% |
| Improvement vs EWM-7 | 11.1% |

### Key Challenges & Solutions
- **90.3% data sparsity**: Handled with binary classifier stage (demand yes/no before quantity)
- **1129 SKUs**: Per-SKU Prophet with category fallback for <10 demand-days products
- **Intermittent demand**: Combined demand frequency features + days-since-last-demand
- **Outlier sensitivity**: 99th percentile capping per SKU, Prophet calibration factor

---

## Task 1: Warehouse Optimization – Technical Approach

### Storage Assignment (Heuristic-Based)
1. **Segment-aware placement**: HF products → PICKING zone (close to expedition), LF products → RESERVE zone
2. **Scoring function**: `score = -distance_to_expedition * w1 - floor * w2`
3. **Heavy item handling**: Ground floor preference for products >5kg
4. **Stackability check**: Gerbable flag determines vertical stacking

### Picking Route Optimization
1. **Multi-chariot splitting**: Weight capacity 300kg per chariot
2. **Nearest-neighbor routing**: Minimizes total travel distance
3. **Congestion detection**: Warns when >3 picks in same zone
4. **3D Manhattan distance**: Accounts for x, y, z positioning

### Preparation Order
- Priority-based batching with segment awareness
- Multi-chariot load balancing by weight

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Version info |
| GET | `/health` | Health check |
| POST | `/predict` | Single/multi-product forecast |
| POST | `/generate-forecast` | Download forecast CSV |
| POST | `/simulate` | Full simulation (assign → pick → route) |
| POST | `/assign-storage` | Storage assignment for product |
| POST | `/optimize-picking` | Picking route optimization |
| GET | `/warehouse-state` | Current warehouse state |
| POST | `/reset-warehouse` | Reset warehouse state |
| POST | `/explain` | XAI: feature importance for product/date |
| POST | `/preparation-order` | Multi-product preparation batching |
| GET | `/model-info` | Model metadata & performance |

---

## Hardware Requirements

- **Training**: ~16 GB RAM, ~20 min (CPU only, no GPU needed)
- **Inference**: ~2 GB RAM, <1s per product prediction
- **API**: ~2 GB RAM, handles concurrent requests via FastAPI async

---

## Model Hosting

All models are stored as JSON files in `models/` (total < 5 MB).
No external model hosting required — fully self-contained.
