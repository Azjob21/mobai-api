"""
MobAI'26 - Task 2: Demand Forecasting - Inference Script
=========================================================
Standalone script that:
  1. Loads saved XGBoost + Prophet models (from ./models/)
  2. Accepts test data path (CSV with historical demand)
  3. Forecasts demand for a configurable future date range
  4. Exports predictions to CSV in required format:
       Date, id_produit, quantite_demande

Usage:
  python inference_forecast.py --input data/test_demand.csv --output forecast_submission.csv
  python inference_forecast.py --start_date 2026-02-15 --end_date 2026-03-17
  python inference_forecast.py  # uses defaults

Output format matches submission guide exactly:
  Date,id_produit,quantite_demande
  15-02-2026,31554,5746.39
  15-02-2026,31565,3467.92
  ...
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from holidays_dz import get_holiday_features_for_date, HOLIDAY_FEATURE_NAMES

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_models():
    """Load all required models and metadata."""
    print("[1/4] Loading models...")
    t0 = time.time()

    # XGBoost classifier (demand yes/no)
    classifier = xgb.Booster()
    classifier.load_model(str(MODELS_DIR / "xgboost_classifier_model.json"))

    # XGBoost regressor (not primary -- Prophet is used for quantity)
    regressor = xgb.Booster()
    regressor.load_model(str(MODELS_DIR / "xgboost_regression_model.json"))

    # Forecast config
    with open(MODELS_DIR / "forecast_config.json", 'r') as f:
        config = json.load(f)

    # Prophet metadata (per-SKU: mean_yhat, cal_factor, trend_slope)
    prophet_meta = {}
    if (MODELS_DIR / "prophet_meta.json").exists():
        with open(MODELS_DIR / "prophet_meta.json", 'r') as f:
            prophet_meta = json.load(f)

    # Product attributes
    product_attrs = {}
    if (MODELS_DIR / "product_attributes.json").exists():
        with open(MODELS_DIR / "product_attributes.json", 'r') as f:
            product_attrs = json.load(f)

    # Delivery stats
    delivery_stats = {}
    if (MODELS_DIR / "delivery_stats.json").exists():
        with open(MODELS_DIR / "delivery_stats.json", 'r') as f:
            delivery_stats = json.load(f)

    # Category encoding
    cat_encoding = {}
    if (MODELS_DIR / "cat_encoding.json").exists():
        with open(MODELS_DIR / "cat_encoding.json", 'r') as f:
            cat_encoding = json.load(f)

    # Product data
    product_priorities = pd.read_csv(DATA_DIR / "product_priorities.csv")
    product_segments = pd.read_csv(DATA_DIR / "product_segments.csv")

    seg_lookup = dict(zip(product_segments["id_produit"], product_segments["segment"]))
    priority_lookup = product_priorities.set_index("id_produit").to_dict("index")
    all_product_ids = list(product_priorities["id_produit"].unique())

    elapsed = time.time() - t0
    print(f"   Models loaded in {elapsed:.1f}s")
    print(f"   Products: {len(all_product_ids)}")
    print(f"   Prophet models: {len(prophet_meta.get('prophet_models', {}))}")

    return {
        'classifier': classifier,
        'regressor': regressor,
        'config': config,
        'prophet_meta': prophet_meta,
        'product_attrs': product_attrs,
        'delivery_stats': delivery_stats,
        'cat_encoding': cat_encoding,
        'seg_lookup': seg_lookup,
        'priority_lookup': priority_lookup,
        'all_product_ids': all_product_ids,
    }


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_features_for_product(product_id, forecast_date, models):
    """Create full feature vector matching the trained XGBoost model's feature set."""
    config = models['config']
    prophet_meta = models['prophet_meta']
    seg_lookup = models['seg_lookup']
    priority_lookup = models['priority_lookup']
    product_attrs = models['product_attrs']
    delivery_stats = models['delivery_stats']
    cat_encoding = models['cat_encoding']
    feature_cols = config.get('feature_cols_regression', [])

    pid_str = str(product_id)
    seg = seg_lookup.get(product_id, "LF")
    is_hf = 1.0 if seg == "HF" else 0.0

    pm = prophet_meta.get("prophet_models", {}).get(pid_str, {})
    sa = prophet_meta.get("simple_avg", {}).get(pid_str, 0.0)
    prophet_yhat = pm.get("mean_yhat", sa) if pm else sa

    pp = priority_lookup.get(product_id, {})
    p_total = pp.get("total_demand", 0.0)
    p_days = pp.get("demand_days", 0.0)
    p_avg = pp.get("avg_demand", 0.0)
    p_freq = pp.get("demand_frequency", 0.0)
    p_prio = pp.get("priority_score", 0.0)
    p_demand_score = pp.get("demand_score", 0.0)
    p_freq_score = pp.get("frequency_score", 0.0)
    prod_avg = max(p_avg, 1e-6)
    categorie = pp.get("categorie", "UNKNOWN")
    cat_enc = float(cat_encoding.get(str(categorie), 0))

    pa = product_attrs.get(pid_str, {})
    colisage_fardeau = pa.get("colisage_fardeau", 1.0)
    colisage_palette = pa.get("colisage_palette", 1.0)
    volume_pcs = pa.get("volume_pcs", 0.0)
    poids_kg = pa.get("poids_kg", 0.0)
    is_gerbable = pa.get("is_gerbable", 0.0)

    ds = delivery_stats.get(pid_str, {})
    del_total_count = ds.get("del_total_count", 0.0)
    del_total_qty = ds.get("del_total_qty", 0.0)
    del_n_days = ds.get("del_n_days", 0.0)
    del_avg_qty = ds.get("del_avg_qty", 0.0)

    dow = float(forecast_date.weekday())
    month = float(forecast_date.month)
    week = float(forecast_date.isocalendar()[1])
    is_wknd = 1.0 if dow >= 5 else 0.0
    dom = float(forecast_date.day)
    qtr = float((forecast_date.month - 1) // 3 + 1)
    day_of_year = float(forecast_date.timetuple().tm_yday)
    lag_approx = prophet_yhat

    # Pre-compute holiday features for this date
    _hol_dict = get_holiday_features_for_date(forecast_date)

    features = {}
    for col in feature_cols:
        if col in _hol_dict:
            features[col] = _hol_dict[col]
        elif col == "cat_enc":
            features[col] = cat_enc
        elif col == "colisage_fardeau":
            features[col] = colisage_fardeau
        elif col == "colisage_palette":
            features[col] = colisage_palette
        elif col == "volume_pcs":
            features[col] = volume_pcs
        elif col == "poids_kg":
            features[col] = poids_kg
        elif col == "is_gerbable":
            features[col] = is_gerbable
        elif col == "is_hf":
            features[col] = is_hf
        elif col == "dow":
            features[col] = dow
        elif col == "month":
            features[col] = month
        elif col == "week":
            features[col] = week
        elif col == "is_wknd":
            features[col] = is_wknd
        elif col == "dom":
            features[col] = dom
        elif col == "qtr":
            features[col] = qtr
        elif col == "day_idx":
            features[col] = 600.0
        elif col == "day_idx_sq":
            features[col] = 600.0 ** 2 / 1e6
        elif col == "is_month_start":
            features[col] = 1.0 if dom <= 3 else 0.0
        elif col == "is_month_end":
            features[col] = 1.0 if dom >= 28 else 0.0
        elif col == "is_week_start":
            features[col] = 1.0 if dow == 0 else 0.0
        elif col.startswith("fourier_sin_y"):
            k = int(col[-1])
            features[col] = math.sin(2 * math.pi * k * day_of_year / 365.25)
        elif col.startswith("fourier_cos_y"):
            k = int(col[-1])
            features[col] = math.cos(2 * math.pi * k * day_of_year / 365.25)
        elif col.startswith("fourier_sin_w"):
            k = int(col[-1])
            features[col] = math.sin(2 * math.pi * k * dow / 7)
        elif col.startswith("fourier_cos_w"):
            k = int(col[-1])
            features[col] = math.cos(2 * math.pi * k * dow / 7)
        elif col.startswith("fourier_sin_m"):
            k = int(col[-1])
            features[col] = math.sin(2 * math.pi * k * dom / 31)
        elif col.startswith("fourier_cos_m"):
            k = int(col[-1])
            features[col] = math.cos(2 * math.pi * k * dom / 31)
        elif col == "prophet_yhat":
            features[col] = prophet_yhat
        elif col == "prophet_trend":
            features[col] = prophet_yhat
        elif col == "prophet_weekly":
            features[col] = 0.0
        elif col == "prophet_yearly":
            features[col] = 0.0
        elif col == "prophet_ratio":
            features[col] = prophet_yhat / prod_avg
        elif col == "prophet_over_ewm7":
            features[col] = prophet_yhat / (lag_approx + 1e-6)
        elif col == "prophet_over_rmean7":
            features[col] = prophet_yhat / (lag_approx + 1e-6)
        elif col == "prophet_resid_lag1":
            features[col] = 0.0
        elif col == "prophet_resid_rmean7":
            features[col] = 0.0
        elif col == "prophet_trend_norm":
            features[col] = prophet_yhat / (prod_avg + 1e-6)
        elif col == "prophet_weekly_abs":
            features[col] = 0.0
        elif col == "prophet_yearly_abs":
            features[col] = 0.0
        elif col == "prophet_seasonal_str":
            features[col] = 0.0
        elif col == "p_total":
            features[col] = p_total
        elif col == "p_days":
            features[col] = p_days
        elif col == "p_avg":
            features[col] = p_avg
        elif col == "p_freq":
            features[col] = p_freq
        elif col == "p_prio":
            features[col] = p_prio
        elif col == "p_demand_score":
            features[col] = p_demand_score
        elif col == "p_freq_score":
            features[col] = p_freq_score
        elif col == "prod_avg_demand":
            features[col] = p_avg
        elif col == "prod_med_demand":
            features[col] = p_avg
        elif col == "prod_std_demand":
            features[col] = p_avg * 0.5
        elif col == "prod_n_days":
            features[col] = p_days
        elif col == "del_total_count":
            features[col] = del_total_count
        elif col == "del_total_qty":
            features[col] = del_total_qty
        elif col == "del_n_days":
            features[col] = del_n_days
        elif col == "del_avg_qty":
            features[col] = del_avg_qty
        elif col.startswith("del_rolling_"):
            features[col] = 0.0
        elif col.startswith("del_qty_rolling_"):
            features[col] = 0.0
        elif col.startswith("lag_") or col.startswith("lag1_"):
            features[col] = lag_approx if "norm" not in col else lag_approx / prod_avg
        elif col.startswith("rmean_") or col.startswith("rmed_"):
            features[col] = lag_approx if "norm" not in col else lag_approx / prod_avg
        elif col.startswith("rstd_"):
            features[col] = lag_approx * 0.3
        elif col.startswith("rmax_"):
            features[col] = lag_approx * 1.5
        elif col.startswith("rmin_"):
            features[col] = max(lag_approx * 0.5, 0)
        elif col.startswith("rsum_"):
            w = int(col.split("_")[-1])
            features[col] = lag_approx * w
        elif col.startswith("dfreq_"):
            features[col] = p_freq
        elif col == "days_since":
            features[col] = 1.0 / max(p_freq, 0.01)
        elif col == "cv_28":
            features[col] = 0.5
        elif col == "ewm_7":
            features[col] = lag_approx
        elif col == "ewm_28":
            features[col] = lag_approx
        elif col == "ewm7_norm":
            features[col] = lag_approx / prod_avg
        elif col == "ewm28_norm":
            features[col] = lag_approx / prod_avg
        elif col == "rmean7_over_28":
            features[col] = 1.0
        elif col == "rmean7_norm":
            features[col] = lag_approx / prod_avg
        elif col == "rmean28_norm":
            features[col] = lag_approx / prod_avg
        elif col == "rmean_wday4_norm":
            features[col] = lag_approx / prod_avg
        elif col.startswith("hf_x_"):
            base_col = col.replace("hf_x_", "")
            base_val = features.get(base_col, lag_approx if "lag" in base_col or "rmean" in base_col else 0.0)
            features[col] = is_hf * base_val
        else:
            features[col] = 0.0

    return pd.DataFrame([features])[feature_cols]


# ============================================================================
# PREDICTION
# ============================================================================

def predict_single_product(product_id, forecast_date, models):
    """Predict demand for a single product on a given date."""
    config = models['config']
    classifier = models['classifier']
    prophet_meta = models['prophet_meta']
    seg_lookup = models['seg_lookup']

    OPTIMAL_THRESHOLD = config['optimal_threshold']
    BIAS_MULTIPLIER = config.get('bias_multiplier', 1.0)
    ENSEMBLE_ALPHA = config.get('ensemble_alpha', 0.7)
    USE_EXPECTED_VALUE = config.get('use_expected_value', False)
    PROB_POWER = config.get('prob_power', 1.0)

    pid_str = str(product_id)
    pm = prophet_meta.get("prophet_models", {}).get(pid_str, {})
    sa = float(prophet_meta.get("simple_avg", {}).get(pid_str, 0))
    prophet_yhat = pm.get("mean_yhat", sa) if pm else sa
    seg = seg_lookup.get(product_id, "LF")

    features_df = create_features_for_product(product_id, forecast_date, models)
    dmatrix = xgb.DMatrix(features_df)
    probability = float(classifier.predict(dmatrix)[0])

    # XGBoost regressor prediction (log1p-transformed)
    regressor = models['regressor']
    reg_pred = float(np.expm1(regressor.predict(dmatrix)[0]))
    reg_pred = max(0, reg_pred)
    # Blend: alpha * Prophet + (1-alpha) * Regressor
    blended_qty = (
        ENSEMBLE_ALPHA * prophet_yhat +
        (1 - ENSEMBLE_ALPHA) * reg_pred
    )

    if USE_EXPECTED_VALUE:
        predicted_demand = (probability ** PROB_POWER) * blended_qty * BIAS_MULTIPLIER
        predicted_demand = max(0, predicted_demand)
    elif probability > OPTIMAL_THRESHOLD:
        predicted_demand = blended_qty * BIAS_MULTIPLIER
        predicted_demand = max(0, predicted_demand)
    else:
        predicted_demand = 0.0

    return round(predicted_demand, 2)


# ============================================================================
# MAIN INFERENCE
# ============================================================================

def run_inference(start_date, end_date, output_file, models):
    """Generate forecast CSV for all products over the date range."""
    print(f"[3/4] Running inference: {start_date} -> {end_date}")

    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)

    all_products = models['all_product_ids']
    total_predictions = len(dates) * len(all_products)
    print(f"   {len(all_products)} products x {len(dates)} days = {total_predictions:,} predictions")

    rows = []
    non_zero = 0
    total_qty = 0.0
    t0 = time.time()

    for d_idx, d in enumerate(dates):
        for pid in all_products:
            qty = predict_single_product(pid, d, models)
            rows.append({
                'Date': d.strftime("%d-%m-%Y"),
                'id_produit': pid,
                'quantite_demande': qty
            })
            if qty > 0:
                non_zero += 1
            total_qty += qty

        elapsed = time.time() - t0
        pct = (d_idx + 1) / len(dates) * 100
        rate = (d_idx + 1) * len(all_products) / elapsed
        eta = (total_predictions - (d_idx + 1) * len(all_products)) / rate if rate > 0 else 0
        print(f"   Day {d_idx+1}/{len(dates)} ({pct:.0f}%) - "
              f"{rate:.0f} pred/s - ETA: {eta:.0f}s", end='\r')

    print()
    elapsed = time.time() - t0

    # Write output CSV
    print(f"[4/4] Writing output to {output_file}...")
    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_file, index=False)

    # Summary
    print(f"\n{'='*60}")
    print(f"  FORECAST COMPLETE")
    print(f"{'='*60}")
    print(f"  Output file:       {output_file}")
    print(f"  Date range:        {start_date.strftime('%d-%m-%Y')} -> {end_date.strftime('%d-%m-%Y')}")
    print(f"  Total products:    {len(all_products)}")
    print(f"  Total days:        {len(dates)}")
    print(f"  Total rows:        {len(rows):,}")
    print(f"  Non-zero forecasts:{non_zero:,} ({non_zero/len(rows)*100:.1f}%)")
    print(f"  Total quantity:    {total_qty:,.0f}")
    print(f"  Runtime:           {elapsed:.1f}s")
    print(f"{'='*60}")

    # Show sample
    print("\n  Sample output (first 5 rows):")
    for row in rows[:5]:
        print(f"    {row['Date']}, {row['id_produit']}, {row['quantite_demande']}")
    print(f"  ...")
    for row in rows[-3:]:
        print(f"    {row['Date']}, {row['id_produit']}, {row['quantite_demande']}")

    return df_out


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MobAI'26 Task 2 - Demand Forecasting Inference")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to input historical demand CSV (optional, used for context)")
    parser.add_argument("--output", type=str, default="forecast_submission.csv",
                        help="Path for output CSV (default: forecast_submission.csv)")
    parser.add_argument("--start_date", type=str, default="2026-02-15",
                        help="Forecast start date YYYY-MM-DD (default: 2026-02-15)")
    parser.add_argument("--end_date", type=str, default="2026-03-17",
                        help="Forecast end date YYYY-MM-DD (default: 2026-03-17)")
    args = parser.parse_args()

    print("=" * 60)
    print("  MobAI'26 - Task 2: Demand Forecasting Inference")
    print("=" * 60)

    # Parse dates
    start_dt = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end_date, "%Y-%m-%d")

    # If input file provided, display info about it
    if args.input:
        print(f"\n[0/4] Reading input data from {args.input}...")
        try:
            input_df = pd.read_csv(args.input)
            print(f"   Input shape: {input_df.shape}")
            print(f"   Columns: {list(input_df.columns)}")
            if 'id_produit' in input_df.columns:
                print(f"   Products in input: {input_df['id_produit'].nunique()}")
        except Exception as e:
            print(f"   Warning: Could not read input file: {e}")
            print(f"   Proceeding with model-based inference for all known products")

    # Load models
    models = load_models()

    # Run inference
    df_result = run_inference(start_dt, end_dt, args.output, models)

    print(f"\nDone! Output saved to: {args.output}")
