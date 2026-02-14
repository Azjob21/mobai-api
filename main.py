from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime, date, timedelta
import json
from pathlib import Path
import math
from holidays_dz import get_holiday_features_for_date, HOLIDAY_FEATURE_NAMES

# Initialize FastAPI
app = FastAPI(
    title="MobAI WMS API",
    description="AI-powered Warehouse Management System API – Forecasting, Storage Optimization, Picking & Simulation",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# LOAD MODELS & DATA
# ============================================================================

print("Loading models and data...")

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Load XGBoost models
classifier = xgb.Booster()
classifier.load_model(str(MODELS_DIR / "xgboost_classifier_model.json"))

regressor = xgb.Booster()
regressor.load_model(str(MODELS_DIR / "xgboost_regression_model.json"))

# Load LF regressor if available
lf_regressor = None
if (MODELS_DIR / "xgboost_lf_regressor_model.json").exists():
    lf_regressor = xgb.Booster()
    lf_regressor.load_model(str(MODELS_DIR / "xgboost_lf_regressor_model.json"))

# Load config
with open(MODELS_DIR / "forecast_config.json", 'r') as f:
    config = json.load(f)

OPTIMAL_THRESHOLD = config['optimal_threshold']
BIAS_MULTIPLIER = config.get('bias_multiplier', 1.0)
ENSEMBLE_ALPHA = config.get('ensemble_alpha', 0.7)
SMEARING_FACTOR = config.get('smearing_factor', 1.0)
HAS_LF_REGRESSOR = config.get('has_lf_regressor', False)
USE_BASE_MARGIN = config.get('use_base_margin', False)
USE_EXPECTED_VALUE = config.get('use_expected_value', False)
PROB_POWER = config.get('prob_power', 1.0)

FEATURE_COLS = config.get('feature_cols_regression', [])

# Load Prophet metadata
prophet_meta = {}
if (MODELS_DIR / "prophet_meta.json").exists():
    with open(MODELS_DIR / "prophet_meta.json", 'r') as f:
        prophet_meta = json.load(f)

# Load product attributes
product_attrs = {}
if (MODELS_DIR / "product_attributes.json").exists():
    with open(MODELS_DIR / "product_attributes.json", 'r') as f:
        product_attrs = json.load(f)

# Load delivery stats
delivery_stats = {}
if (MODELS_DIR / "delivery_stats.json").exists():
    with open(MODELS_DIR / "delivery_stats.json", 'r') as f:
        delivery_stats = json.load(f)

# Load category encoding
cat_encoding = {}
if (MODELS_DIR / "cat_encoding.json").exists():
    with open(MODELS_DIR / "cat_encoding.json", 'r') as f:
        cat_encoding = json.load(f)

# Load data
product_priorities = pd.read_csv(DATA_DIR / "product_priorities.csv")
product_segments = pd.read_csv(DATA_DIR / "product_segments.csv")
warehouse_locations = pd.read_csv(DATA_DIR / "warehouse_locations.csv")

# Build lookups
seg_lookup = dict(zip(product_segments["id_produit"], product_segments["segment"]))
priority_lookup = product_priorities.set_index("id_produit").to_dict("index")
all_product_ids = list(product_priorities["id_produit"].unique())

# Zone definitions
RECEIPT_ZONE = {'x': 0, 'y': 0, 'z': 0}
EXPEDITION_ZONE = {'x': 3, 'y': 5, 'z': 0}

# Chariot configuration
CHARIOT_CAPACITY_KG = 300.0
CHARIOT_MAX_ITEMS = 20

print("Models and data loaded!")

# ============================================================================
# WAREHOUSE STATE MANAGER
# ============================================================================

class WarehouseState:
    """In-memory warehouse state tracking slot occupancy."""

    def __init__(self, locations_df: pd.DataFrame):
        self.locations_df = locations_df.copy()
        self.occupied_slots: Dict[int, Dict] = {}
        self.product_locations: Dict[int, List[int]] = {}
        self.loc_by_id: Dict[int, dict] = {}
        for _, row in locations_df.iterrows():
            self.loc_by_id[int(row['id_emplacement'])] = row.to_dict()

    def get_available_slots(self, slot_type: str = None) -> pd.DataFrame:
        occupied_ids = set(self.occupied_slots.keys())
        available = self.locations_df[~self.locations_df['id_emplacement'].isin(occupied_ids)]
        if slot_type:
            available = available[available['type_emplacement'] == slot_type]
        return available

    def assign_slot(self, slot_id: int, product_id: int, quantity: int, timestamp: str = None):
        self.occupied_slots[slot_id] = {
            'product_id': product_id,
            'quantity': quantity,
            'timestamp': timestamp or datetime.now().isoformat()
        }
        if product_id not in self.product_locations:
            self.product_locations[product_id] = []
        if slot_id not in self.product_locations[product_id]:
            self.product_locations[product_id].append(slot_id)

    def release_slot(self, slot_id: int):
        if slot_id in self.occupied_slots:
            pid = self.occupied_slots[slot_id]['product_id']
            del self.occupied_slots[slot_id]
            if pid in self.product_locations:
                self.product_locations[pid] = [s for s in self.product_locations[pid] if s != slot_id]
                if not self.product_locations[pid]:
                    del self.product_locations[pid]

    def find_product_slots(self, product_id: int) -> List[int]:
        return self.product_locations.get(product_id, [])

    def get_occupancy_stats(self) -> dict:
        total = len(self.locations_df)
        occupied = len(self.occupied_slots)
        picking_total = len(self.locations_df[self.locations_df['type_emplacement'] == 'PICKING'])
        reserve_total = len(self.locations_df[self.locations_df['type_emplacement'] == 'RESERVE'])
        picking_occ = sum(1 for sid in self.occupied_slots
                          if self.loc_by_id.get(sid, {}).get('type_emplacement') == 'PICKING')
        reserve_occ = sum(1 for sid in self.occupied_slots
                          if self.loc_by_id.get(sid, {}).get('type_emplacement') == 'RESERVE')
        return {
            'total_slots': total,
            'occupied_slots': occupied,
            'available_slots': total - occupied,
            'occupancy_rate': round(occupied / total * 100, 1) if total > 0 else 0,
            'picking': {'total': picking_total, 'occupied': picking_occ,
                        'available': picking_total - picking_occ},
            'reserve': {'total': reserve_total, 'occupied': reserve_occ,
                        'available': reserve_total - reserve_occ}
        }

    def reset(self):
        self.occupied_slots.clear()
        self.product_locations.clear()


# Initialize global warehouse state
warehouse_state = WarehouseState(warehouse_locations)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ForecastRequest(BaseModel):
    product_ids: List[int]
    date: date

class ForecastResponse(BaseModel):
    product_id: int
    forecast_date: str
    predicted_demand: float
    probability: float
    confidence_low: float = 0.0
    confidence_high: float = 0.0
    segment: str = "LF"
    explanation: Optional[Dict[str, Any]] = None

class StorageRequest(BaseModel):
    product_id: int
    quantity: int
    weight_kg: Optional[float] = None

class StorageResponse(BaseModel):
    product_id: int
    assigned_location: str
    location_id: int
    zone: str = ""
    distance_to_expedition: float
    floor: int
    priority_score: float
    reasoning: str = ""
    route_from_receipt: Optional[str] = None

class PickingItem(BaseModel):
    product_id: int
    quantity: int
    location_id: Optional[int] = None

class PickingRequest(BaseModel):
    items: List[PickingItem]
    chariot_capacity_kg: float = CHARIOT_CAPACITY_KG
    max_chariots: int = 3

class PickingRouteStep(BaseModel):
    step: int
    product_id: Optional[int] = None
    location_code: str
    zone: str = ""
    floor: int = 0
    distance_from_previous: float = 0.0
    action: str = "pick"
    detail: str = ""

class ChariotRoute(BaseModel):
    chariot_id: int
    items_count: int
    total_weight_kg: float
    route: List[PickingRouteStep]
    total_distance: float

class PickingResponse(BaseModel):
    chariots: List[ChariotRoute]
    total_distance: float
    total_items: int
    total_chariots_used: int
    efficiency_improvement: float
    congestion_warnings: List[str] = []

class SimulationEvent(BaseModel):
    date: str
    product_id: int
    quantity: int
    flow_type: str = Field(..., description="'ingoing' or 'outgoing'")

class SimulationRequest(BaseModel):
    events: List[SimulationEvent]
    reset_state: bool = True

class SimulationAction(BaseModel):
    event_index: int
    date: str
    product_id: int
    flow_type: str
    quantity: int
    action: str
    location_code: str = ""
    location_id: int = 0
    zone: str = ""
    route_description: str = ""
    chariot_trips: int = 1
    reasoning: str = ""
    forecast_demand: Optional[float] = None
    success: bool = True
    warning: str = ""

class SimulationResponse(BaseModel):
    total_events_processed: int
    actions: List[SimulationAction]
    final_warehouse_state: dict
    congestion_log: List[str] = []
    assumptions: List[str] = []

class ForecastGenerateRequest(BaseModel):
    start_date: str = "2026-01-09"
    end_date: str = "2026-02-08"
    output_file: str = "forecast_submission.csv"

class ForecastGenerateResponse(BaseModel):
    total_products: int
    total_days: int
    total_rows: int
    non_zero_forecasts: int
    total_predicted_quantity: float
    file_path: str
    sample_rows: List[dict]

class XAIRequest(BaseModel):
    product_id: int
    date: date

class XAIResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    product_id: int
    segment: str
    prophet_baseline: float
    classifier_probability: float
    threshold: float
    final_prediction: float
    confidence_interval: dict
    top_factors: List[dict]
    model_components: dict
    assumptions: List[str]

class WarehouseStateResponse(BaseModel):
    occupancy: dict
    product_count: int
    occupied_slots_detail: List[dict]

class PreparationOrderRequest(BaseModel):
    date: date
    override_quantities: Optional[Dict[int, float]] = None

class PreparationOrderItem(BaseModel):
    product_id: int
    predicted_demand: float
    override_applied: bool = False
    source_location: str = ""
    source_location_id: int = 0
    quantity_to_prepare: float = 0.0
    status: str = "pending"

class PreparationOrderResponse(BaseModel):
    date: str
    total_items: int
    items: List[PreparationOrderItem]
    picking_route: Optional[PickingResponse] = None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_features_for_product(product_id: int, forecast_date: datetime) -> pd.DataFrame:
    """Create full feature vector matching the trained model's feature set."""
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
    cat_enc = float(cat_encoding.get(categorie, 0))

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
    for col in FEATURE_COLS:
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

    return pd.DataFrame([features])[FEATURE_COLS]


def predict_single_product(product_id: int, forecast_date: datetime) -> dict:
    """Predict demand for a single product on a given date."""
    pid_str = str(product_id)
    pm = prophet_meta.get("prophet_models", {}).get(pid_str, {})
    sa = float(prophet_meta.get("simple_avg", {}).get(pid_str, 0))
    prophet_yhat = pm.get("mean_yhat", sa) if pm else sa
    cal_factor = pm.get("cal_factor", 1.0) if pm else 1.0
    seg = seg_lookup.get(product_id, "LF")

    features_df = create_features_for_product(product_id, forecast_date)
    dmatrix = xgb.DMatrix(features_df)
    probability = float(classifier.predict(dmatrix)[0])

    # XGBoost regressor prediction (log1p-transformed, need expm1)
    reg_pred = float(np.expm1(regressor.predict(dmatrix)[0]))
    reg_pred = max(0, reg_pred)
    # Blend: alpha * Prophet + (1-alpha) * Regressor
    blended_qty = (
        ENSEMBLE_ALPHA * prophet_yhat +
        (1 - ENSEMBLE_ALPHA) * reg_pred
    )

    if USE_EXPECTED_VALUE:
        # Expected-value mode: pred = prob^power * blend * bias
        predicted_demand = (probability ** PROB_POWER) * blended_qty * BIAS_MULTIPLIER
        predicted_demand = max(0, predicted_demand)
    elif probability > OPTIMAL_THRESHOLD:
        predicted_demand = blended_qty * BIAS_MULTIPLIER
        predicted_demand = max(0, predicted_demand)
    else:
        predicted_demand = 0.0

    pp = priority_lookup.get(product_id, {})
    p_avg = pp.get("avg_demand", 0.0)
    std_est = p_avg * 0.4 if p_avg > 0 else predicted_demand * 0.5
    confidence_low = max(0, predicted_demand - 1.96 * std_est)
    confidence_high = max(0, predicted_demand + 1.96 * std_est)

    return {
        'product_id': product_id,
        'predicted_demand': round(predicted_demand, 2),
        'probability': round(probability, 4),
        'confidence_low': round(confidence_low, 2),
        'confidence_high': round(confidence_high, 2),
        'segment': seg,
        'prophet_yhat': prophet_yhat,
        'cal_factor': cal_factor,
    }


def manhattan_distance_3d(x1, y1, z1, x2, y2, z2):
    return abs(x2 - x1) + abs(y2 - y1) + abs(z2 - z1)


def get_zone_description(loc_row: dict) -> str:
    """Generate human-readable zone path description."""
    zone = str(loc_row.get('zone', ''))
    floor_label = str(loc_row.get('floor', ''))
    code = str(loc_row.get('code_emplacement', ''))
    z_val = int(loc_row.get('z', 0))

    parts = []
    if 'B7-PCK' in zone:
        parts.append("Zone Picking B7")
    elif 'B07-N' in zone:
        level = zone.replace('B07-', '')
        parts.append(f"Zone Reserve {level}")
    elif 'B07-SS' in zone:
        parts.append("Zone Sous-sol Reserve")
    else:
        parts.append(f"Zone {zone}")

    if floor_label and floor_label != 'nan':
        corridor = floor_label[:2] if len(floor_label) >= 2 else floor_label
        parts.append(f"Allee {corridor}")

    if z_val > 0:
        parts.append(f"Etage {z_val} (acces lift)")
    else:
        parts.append("RDC")

    parts.append(f"Emplacement {code}")
    return " -> ".join(parts)


def build_route_description(from_pos: dict, to_loc: dict) -> str:
    """Build detailed route narrative from a position to a location."""
    z_from = from_pos.get('z', 0)
    z_to = int(to_loc.get('z', 0))
    code = str(to_loc.get('code_emplacement', ''))
    zone = str(to_loc.get('zone', ''))

    steps = []
    if z_from != z_to:
        if z_to > z_from:
            steps.append(f"Monter au niveau {z_to} via lift")
        else:
            steps.append(f"Descendre au niveau {z_to} via lift")

    floor_label = str(to_loc.get('floor', ''))
    if floor_label and floor_label != 'nan':
        steps.append(f"Allee {floor_label}")

    steps.append(f"Se rendre a {code}")

    if 'PCK' in zone:
        steps.append("(zone picking)")
    elif 'RESERVE' in str(to_loc.get('type_emplacement', '')):
        steps.append("(zone reserve)")

    return " -> ".join(steps)


def assign_storage_for_product(product_id: int, quantity: int,
                                weight_kg: float = None,
                                prefer_picking: bool = False) -> dict:
    """Smart storage assignment with state tracking and demand-aware placement."""
    pp = priority_lookup.get(product_id, {})
    priority_score = pp.get("priority_score", 0.0)
    p_freq = pp.get("demand_frequency", 0.0)
    prod_weight = weight_kg if weight_kg else pp.get("weight", 0.0)
    seg = seg_lookup.get(product_id, "LF")

    if prefer_picking or (seg == "HF" and p_freq > 0.01):
        slot_type = "PICKING"
    else:
        slot_type = "RESERVE"

    available = warehouse_state.get_available_slots(slot_type)
    if len(available) == 0:
        slot_type = "PICKING" if slot_type == "RESERVE" else "RESERVE"
        available = warehouse_state.get_available_slots(slot_type)

    if len(available) == 0:
        return {
            'success': False,
            'location_code': '', 'location_id': 0, 'zone': '', 'floor': 0,
            'dist_to_expedition': 0, 'priority_score': priority_score,
            'reasoning': 'Tous les emplacements sont occupes',
            'route_from_receipt': '', 'zone_desc': ''
        }

    available = available.copy()
    available['dist_exp'] = pd.to_numeric(available['dist_from_expedition'], errors='coerce').fillna(8)
    z_vals = pd.to_numeric(available['z'], errors='coerce').fillna(0)

    if seg == "HF":
        available['score'] = -available['dist_exp'] * 3.0 - z_vals * 5.0
    else:
        available['score'] = -available['dist_exp'] * 1.0 - z_vals * 2.0

    if prod_weight and prod_weight > 5:
        available['score'] = available['score'] - z_vals * 10.0

    best = available.nlargest(1, 'score').iloc[0]
    slot_id = int(best['id_emplacement'])
    warehouse_state.assign_slot(slot_id, product_id, quantity)

    loc_dict = best.to_dict()
    zone_desc = get_zone_description(loc_dict)
    route = build_route_description(RECEIPT_ZONE, loc_dict)

    reasoning_parts = [
        f"Segment={seg}", f"Freq={p_freq:.4f}", f"Type={slot_type}",
        f"Dist exp={best['dist_from_expedition']}", f"Etage={int(best['z'])}",
    ]
    if prod_weight and prod_weight > 5:
        reasoning_parts.append(f"Lourd ({prod_weight}kg) -> sol")

    return {
        'success': True,
        'location_code': best['code_emplacement'],
        'location_id': slot_id,
        'zone': str(best.get('zone', '')),
        'floor': int(best['z']),
        'dist_to_expedition': float(best['dist_from_expedition']),
        'priority_score': priority_score,
        'reasoning': '; '.join(reasoning_parts),
        'route_from_receipt': f"Reception -> {route}",
        'zone_desc': zone_desc
    }


def optimize_picking_route(items_with_locs: pd.DataFrame,
                            chariot_capacity_kg: float = CHARIOT_CAPACITY_KG,
                            max_chariots: int = 3) -> dict:
    """Multi-chariot picking optimization with nearest-neighbor routing."""
    if len(items_with_locs) == 0:
        return {'chariots': [], 'total_distance': 0, 'total_items': 0,
                'total_chariots_used': 0, 'efficiency_improvement': 0,
                'congestion_warnings': []}

    items = items_with_locs.copy()
    items['weight_est'] = items['product_id'].apply(
        lambda pid: priority_lookup.get(pid, {}).get('weight', 0.5)
    )

    # Split into chariot loads
    chariot_loads = []
    current_load = []
    current_weight = 0.0

    for idx, row in items.iterrows():
        item_weight = row.get('weight_est', 0.5) * row.get('quantity', 1)
        if current_weight + item_weight > chariot_capacity_kg and current_load:
            chariot_loads.append(pd.DataFrame(current_load))
            current_load = []
            current_weight = 0.0
            if len(chariot_loads) >= max_chariots - 1:
                remaining_items = items.loc[idx:]
                chariot_loads.append(remaining_items)
                current_load = []
                break
        current_load.append(row)
        current_weight += item_weight

    if current_load:
        chariot_loads.append(pd.DataFrame(current_load))
    if not chariot_loads:
        chariot_loads = [items]

    all_chariots = []
    total_distance = 0
    congestion_warnings = []
    zone_visit_count = {}

    for cidx, load_df in enumerate(chariot_loads):
        current_pos = EXPEDITION_ZONE.copy()
        route = []
        remaining = load_df.copy()
        chariot_dist = 0
        chariot_weight = 0

        while len(remaining) > 0:
            remaining = remaining.copy()
            remaining['dist'] = remaining.apply(
                lambda row: manhattan_distance_3d(
                    current_pos['x'], current_pos['y'], current_pos['z'],
                    row['x'], row['y'], row['z']),
                axis=1)

            nearest_idx = remaining['dist'].idxmin()
            nearest = remaining.loc[nearest_idx]

            loc_code = str(nearest.get('code_emplacement', 'UNKNOWN'))
            loc_zone = str(nearest.get('zone', ''))
            loc_z = int(nearest.get('z', 0))

            zone_key = f"{loc_zone}-{loc_z}"
            zone_visit_count[zone_key] = zone_visit_count.get(zone_key, 0) + 1

            detail = build_route_description(current_pos, nearest.to_dict())

            route.append(PickingRouteStep(
                step=len(route) + 1,
                product_id=int(nearest['product_id']),
                location_code=loc_code,
                zone=loc_zone, floor=loc_z,
                distance_from_previous=float(nearest['dist']),
                action="pick", detail=detail))

            chariot_dist += nearest['dist']
            chariot_weight += nearest.get('weight_est', 0.5) * nearest.get('quantity', 1)
            current_pos = {'x': nearest['x'], 'y': nearest['y'], 'z': int(nearest['z'])}
            remaining = remaining.drop(nearest_idx)

        return_dist = manhattan_distance_3d(
            current_pos['x'], current_pos['y'], current_pos['z'],
            EXPEDITION_ZONE['x'], EXPEDITION_ZONE['y'], EXPEDITION_ZONE['z'])
        route.append(PickingRouteStep(
            step=len(route) + 1, product_id=None,
            location_code="EXPEDITION", zone="B7-N0", floor=0,
            distance_from_previous=float(return_dist),
            action="return", detail="Retour zone expedition"))
        chariot_dist += return_dist

        all_chariots.append(ChariotRoute(
            chariot_id=cidx + 1, items_count=len(load_df),
            total_weight_kg=round(chariot_weight, 1),
            route=route, total_distance=float(chariot_dist)))
        total_distance += chariot_dist

    for zone_key, count in zone_visit_count.items():
        if count > 3:
            congestion_warnings.append(
                f"Zone {zone_key}: {count} picks - risque de congestion")

    naive_distance = len(items) * 16
    eff = ((naive_distance - total_distance) / naive_distance) * 100 if naive_distance > 0 else 0

    return {
        'chariots': all_chariots,
        'total_distance': float(total_distance),
        'total_items': len(items),
        'total_chariots_used': len(all_chariots),
        'efficiency_improvement': float(eff),
        'congestion_warnings': congestion_warnings
    }


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = BASE_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"), status_code=200)
    return HTMLResponse(content="<h1>MobAI'26 API</h1><p>Visit <a href='/docs'>/docs</a> for Swagger UI</p>", status_code=200)


@app.get("/api")
async def api_info():
    return {
        "service": "MobAI WMS API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "forecasting": "/predict",
            "forecast_csv": "/generate-forecast",
            "storage": "/assign-storage",
            "picking": "/optimize-picking",
            "simulation": "/simulate",
            "warehouse_state": "/warehouse-state",
            "explainability": "/explain",
            "preparation_order": "/preparation-order",
            "reset_warehouse": "/reset-warehouse",
            "model_info": "/model-info"
        }
    }


@app.get("/health")
async def health_check():
    occupancy = warehouse_state.get_occupancy_stats()
    return {
        "status": "healthy",
        "models": {
            "classifier": "loaded",
            "regressor": "loaded",
            "prophet_meta": f"{len(prophet_meta.get('prophet_models', {}))} products"
        },
        "data": {
            "products": len(product_priorities),
            "locations": len(warehouse_locations),
            "segments": {"HF": sum(1 for v in seg_lookup.values() if v == "HF"),
                         "LF": sum(1 for v in seg_lookup.values() if v == "LF")}
        },
        "warehouse_occupancy": occupancy,
        "config": {
            "threshold": OPTIMAL_THRESHOLD,
            "bias_multiplier": BIAS_MULTIPLIER,
            "ensemble_alpha": ENSEMBLE_ALPHA,
            "wape": config.get("performance", {}).get("wape_demand_days", "N/A"),
            "classifier_auc": config.get("performance", {}).get("classifier_auc", "N/A")
        }
    }


# ============================================================================
# PREDICT (enhanced with confidence + explanation)
# ============================================================================

@app.post("/predict", response_model=List[ForecastResponse])
async def predict_demand(request: ForecastRequest):
    try:
        forecast_date = datetime.combine(request.date, datetime.min.time())
        results = []
        for product_id in request.product_ids:
            pred = predict_single_product(product_id, forecast_date)
            explanation = {
                "prophet_baseline": round(pred['prophet_yhat'], 2),
                "classifier_probability": pred['probability'],
                "threshold_used": OPTIMAL_THRESHOLD,
                "ensemble_alpha": ENSEMBLE_ALPHA,
                "bias_multiplier": BIAS_MULTIPLIER,
                "method": "Prophet-direct blend with XGBoost classifier gate",
                "decision": "demand predicted" if pred['probability'] > OPTIMAL_THRESHOLD else "no demand (below threshold)"
            }
            results.append(ForecastResponse(
                product_id=pred['product_id'],
                forecast_date=str(request.date),
                predicted_demand=pred['predicted_demand'],
                probability=pred['probability'],
                confidence_low=pred['confidence_low'],
                confidence_high=pred['confidence_high'],
                segment=pred['segment'],
                explanation=explanation))
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ============================================================================
# FORECAST CSV GENERATION
# ============================================================================

@app.post("/generate-forecast", response_model=ForecastGenerateResponse)
async def generate_forecast(request: ForecastGenerateRequest):
    """Generate forecast submission CSV: date, id_produit, quantite_demande."""
    try:
        start = datetime.strptime(request.start_date, "%Y-%m-%d")
        end = datetime.strptime(request.end_date, "%Y-%m-%d")

        dates = []
        current = start
        while current <= end:
            dates.append(current)
            current += timedelta(days=1)

        rows = []
        non_zero = 0
        total_qty = 0.0

        for d in dates:
            for pid in all_product_ids:
                pred = predict_single_product(pid, d)
                qty = round(pred['predicted_demand'], 2)
                rows.append({
                    'date': d.strftime("%Y-%m-%d"),
                    'id_produit': pid,
                    'quantite_demande': qty
                })
                if qty > 0:
                    non_zero += 1
                total_qty += qty

        output_path = BASE_DIR / request.output_file
        df_out = pd.DataFrame(rows)
        df_out.to_csv(output_path, index=False)

        sample = rows[:5] + rows[-5:] if len(rows) > 10 else rows

        return ForecastGenerateResponse(
            total_products=len(all_product_ids),
            total_days=len(dates),
            total_rows=len(rows),
            non_zero_forecasts=non_zero,
            total_predicted_quantity=round(total_qty, 2),
            file_path=str(output_path),
            sample_rows=sample)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast generation error: {str(e)}")


# ============================================================================
# SIMULATION ENDPOINT
# ============================================================================

@app.post("/simulate", response_model=SimulationResponse)
async def simulate(request: SimulationRequest):
    """Process chronological sequence of ingoing/outgoing product events.

    - ingoing: product arrives -> assigned to optimal storage location
    - outgoing: product leaves -> picked from stored location, routing generated
    """
    try:
        if request.reset_state:
            warehouse_state.reset()

        actions = []
        congestion_log = []
        daily_zone_visits = {}

        for idx, event in enumerate(request.events):
            event_date = event.date
            pid = event.product_id
            qty = event.quantity
            flow = event.flow_type.lower()

            try:
                dt = datetime.strptime(event_date, "%Y-%m-%d")
                forecast = predict_single_product(pid, dt)
                forecast_demand = forecast['predicted_demand']
            except Exception:
                forecast_demand = None

            pp = priority_lookup.get(pid, {})
            prod_weight = pp.get("weight", 0.0)
            seg = seg_lookup.get(pid, "LF")

            if flow == "ingoing":
                result = assign_storage_for_product(pid, qty, weight_kg=prod_weight)
                if result['success']:
                    item_weight = (prod_weight or 0.5) * qty
                    chariot_trips = max(1, math.ceil(item_weight / CHARIOT_CAPACITY_KG))
                    actions.append(SimulationAction(
                        event_index=idx, date=event_date, product_id=pid,
                        flow_type="ingoing", quantity=qty,
                        action=f"Reception -> Stockage en {result['location_code']}",
                        location_code=result['location_code'],
                        location_id=result['location_id'],
                        zone=result['zone'],
                        route_description=result['route_from_receipt'],
                        chariot_trips=chariot_trips,
                        reasoning=result['reasoning'],
                        forecast_demand=forecast_demand,
                        success=True, warning=""))
                else:
                    actions.append(SimulationAction(
                        event_index=idx, date=event_date, product_id=pid,
                        flow_type="ingoing", quantity=qty,
                        action="Stockage impossible - pas d'emplacement disponible",
                        reasoning=result['reasoning'],
                        forecast_demand=forecast_demand,
                        success=False, warning="Entrepot plein"))

            elif flow == "outgoing":
                stored_slots = warehouse_state.find_product_slots(pid)
                if stored_slots:
                    slot_id = stored_slots[0]
                    slot_info = warehouse_state.loc_by_id.get(slot_id, {})
                    loc_code = str(slot_info.get('code_emplacement', 'UNKNOWN'))
                    loc_zone = str(slot_info.get('zone', ''))

                    route_desc = build_route_description(EXPEDITION_ZONE, slot_info)
                    full_route = f"Expedition -> {route_desc} -> Retour Expedition"

                    day_zone_key = f"{event_date}_{loc_zone}"
                    daily_zone_visits[day_zone_key] = daily_zone_visits.get(day_zone_key, 0) + 1
                    if daily_zone_visits[day_zone_key] > 5:
                        cw = f"{event_date}: Zone {loc_zone} - {daily_zone_visits[day_zone_key]} mouvements, congestion possible"
                        if cw not in congestion_log:
                            congestion_log.append(cw)

                    warehouse_state.release_slot(slot_id)

                    item_weight = (prod_weight or 0.5) * qty
                    chariot_trips = max(1, math.ceil(item_weight / CHARIOT_CAPACITY_KG))

                    actions.append(SimulationAction(
                        event_index=idx, date=event_date, product_id=pid,
                        flow_type="outgoing", quantity=qty,
                        action=f"Picking depuis {loc_code} -> Expedition",
                        location_code=loc_code, location_id=slot_id,
                        zone=loc_zone,
                        route_description=full_route,
                        chariot_trips=chariot_trips,
                        reasoning=f"Produit trouve en {loc_code} (segment={seg}, freq={pp.get('demand_frequency', 0):.4f})",
                        forecast_demand=forecast_demand,
                        success=True, warning=""))
                else:
                    actions.append(SimulationAction(
                        event_index=idx, date=event_date, product_id=pid,
                        flow_type="outgoing", quantity=qty,
                        action="Produit non stocke - rupture de stock",
                        reasoning=f"Aucun emplacement trouve pour produit {pid}. Verifier approvisionnements.",
                        forecast_demand=forecast_demand,
                        success=False, warning="Stock insuffisant"))
            else:
                actions.append(SimulationAction(
                    event_index=idx, date=event_date, product_id=pid,
                    flow_type=flow, quantity=qty,
                    action="Type de flux inconnu",
                    success=False,
                    warning=f"flow_type doit etre 'ingoing' ou 'outgoing', recu: '{flow}'"))

        final_state = warehouse_state.get_occupancy_stats()

        assumptions = [
            "Chaque emplacement stocke un seul produit a la fois (pas de multi-reference)",
            "Les produits HF sont prioritairement places en PICKING (proche expedition)",
            "Les produits LF sont places en RESERVE",
            f"Capacite chariot : {CHARIOT_CAPACITY_KG} kg max",
            "Routage par nearest-neighbor (distance Manhattan 3D)",
            "Les poids produits non renseignes sont estimes a 0.5 kg",
            "Les evenements sont traites dans l'ordre chronologique fourni",
            "Le lift ajoute un temps fixe de 30s par changement d'etage"
        ]

        return SimulationResponse(
            total_events_processed=len(request.events),
            actions=actions,
            final_warehouse_state=final_state,
            congestion_log=congestion_log,
            assumptions=assumptions)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}")


# ============================================================================
# ENHANCED STORAGE
# ============================================================================

@app.post("/assign-storage", response_model=StorageResponse)
async def assign_storage(request: StorageRequest):
    try:
        result = assign_storage_for_product(
            request.product_id, request.quantity, weight_kg=request.weight_kg)
        if not result['success']:
            raise HTTPException(status_code=404, detail=result['reasoning'])
        return StorageResponse(
            product_id=request.product_id,
            assigned_location=result['location_code'],
            location_id=result['location_id'],
            zone=result['zone'],
            distance_to_expedition=result['dist_to_expedition'],
            floor=result['floor'],
            priority_score=result['priority_score'],
            reasoning=result['reasoning'],
            route_from_receipt=result['route_from_receipt'])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Storage error: {str(e)}")


# ============================================================================
# ENHANCED PICKING (multi-chariot, congestion, enriched route)
# ============================================================================

@app.post("/optimize-picking", response_model=PickingResponse)
async def optimize_picking(request: PickingRequest):
    try:
        if len(request.items) == 0:
            raise HTTPException(status_code=400, detail="Empty picking list")

        items_data = []
        for item in request.items:
            loc_id = item.location_id
            if loc_id is None or loc_id == 0:
                stored = warehouse_state.find_product_slots(item.product_id)
                if stored:
                    loc_id = stored[0]
                else:
                    matching = warehouse_locations[
                        warehouse_locations['type_emplacement'] == 'PICKING'].head(1)
                    if len(matching) > 0:
                        loc_id = int(matching.iloc[0]['id_emplacement'])
            items_data.append({
                'product_id': item.product_id,
                'quantity': item.quantity,
                'location_id': loc_id or 0})

        items_df = pd.DataFrame(items_data)
        items_with_locs = items_df.merge(
            warehouse_locations[['id_emplacement', 'code_emplacement', 'x', 'y', 'z', 'zone', 'floor']],
            left_on='location_id', right_on='id_emplacement', how='left')

        if items_with_locs['x'].isna().any():
            raise HTTPException(status_code=404, detail="Some locations not found")

        result = optimize_picking_route(
            items_with_locs,
            chariot_capacity_kg=request.chariot_capacity_kg,
            max_chariots=request.max_chariots)
        return PickingResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Picking error: {str(e)}")


# ============================================================================
# WAREHOUSE STATE
# ============================================================================

@app.get("/warehouse-state", response_model=WarehouseStateResponse)
async def get_warehouse_state():
    """Get current warehouse state: occupancy, stored products."""
    occupancy = warehouse_state.get_occupancy_stats()
    detail = []
    for slot_id, info in warehouse_state.occupied_slots.items():
        loc = warehouse_state.loc_by_id.get(slot_id, {})
        detail.append({
            'slot_id': slot_id,
            'code': loc.get('code_emplacement', ''),
            'zone': loc.get('zone', ''),
            'type': loc.get('type_emplacement', ''),
            'product_id': info['product_id'],
            'quantity': info['quantity'],
            'stored_at': info['timestamp']})
    return WarehouseStateResponse(
        occupancy=occupancy,
        product_count=len(warehouse_state.product_locations),
        occupied_slots_detail=detail)


@app.post("/reset-warehouse")
async def reset_warehouse():
    """Reset warehouse state to empty."""
    warehouse_state.reset()
    return {"status": "reset", "message": "Warehouse state cleared"}


# ============================================================================
# XAI / EXPLAINABILITY
# ============================================================================

@app.post("/explain", response_model=XAIResponse)
async def explain_forecast(request: XAIRequest):
    """Explain a forecast prediction with feature importance and reasoning."""
    try:
        forecast_date = datetime.combine(request.date, datetime.min.time())
        pid = request.product_id
        pid_str = str(pid)

        pred = predict_single_product(pid, forecast_date)

        pm = prophet_meta.get("prophet_models", {}).get(pid_str, {})
        sa = float(prophet_meta.get("simple_avg", {}).get(pid_str, 0))
        prophet_yhat = pm.get("mean_yhat", sa) if pm else sa
        trend_slope = pm.get("trend_slope", 0.0)
        cal_factor = pm.get("cal_factor", 1.0)

        pp = priority_lookup.get(pid, {})
        seg = seg_lookup.get(pid, "LF")

        top_factors = [
            {"factor": "Prophet baseline (mean_yhat)", "value": round(prophet_yhat, 2),
             "impact": "high", "description": "Prediction de base Prophet par produit"},
            {"factor": "Classifier probability", "value": pred['probability'],
             "impact": "high" if pred['probability'] > 0.5 else "medium",
             "description": f"Probabilite de demande > seuil ({OPTIMAL_THRESHOLD})"},
            {"factor": "Demand frequency", "value": round(pp.get('demand_frequency', 0), 4),
             "impact": "high", "description": "Frequence historique de demande"},
            {"factor": "Segment", "value": seg,
             "impact": "medium", "description": "HF = haute frequence, LF = basse frequence"},
            {"factor": "Avg historical demand", "value": round(pp.get('avg_demand', 0), 2),
             "impact": "medium", "description": "Demande moyenne historique les jours de commande"},
            {"factor": "Calendar (day_of_week)", "value": int(forecast_date.weekday()),
             "impact": "low", "description": "Jour de la semaine (0=lundi)"},
            {"factor": "Trend slope", "value": round(trend_slope, 4),
             "impact": "low", "description": "Pente tendance Prophet"},
            {"factor": "Calibration factor", "value": round(cal_factor, 3),
             "impact": "medium", "description": "Facteur calibration Prophet vs reel"},
        ]

        model_components = {
            "prophet": {
                "mean_yhat": round(prophet_yhat, 2),
                "trend_slope": round(trend_slope, 4),
                "calibration_factor": round(cal_factor, 3),
                "description": "Prophet per-SKU: cps=0.01, sps=0.1, multiplicative, 5 regresseurs temporels"
            },
            "classifier": {
                "model": "XGBoost binaire",
                "probability": pred['probability'],
                "threshold": OPTIMAL_THRESHOLD,
                "auc": config.get("performance", {}).get("classifier_auc", "N/A"),
                "description": "Classifier binaire pour filtrer les jours sans demande"
            },
            "ensemble": {
                "method": "Prophet-direct blend",
                "alpha": ENSEMBLE_ALPHA,
                "bias_multiplier": BIAS_MULTIPLIER,
                "formula": f"({ENSEMBLE_ALPHA} * prophet_yhat + {1-ENSEMBLE_ALPHA} * ewm) * {BIAS_MULTIPLIER}"
            }
        }

        assumptions = [
            "Prophet entraine sur historique 2024-03-02 a 2026-01-08",
            "Classifier XGBoost filtre les produits sans demande (seuil optimal F1-score)",
            "Prediction finale = Prophet-direct avec calibration par produit",
            f"WAPE={config.get('performance', {}).get('wape_demand_days', 'N/A')}%, "
            f"Bias={config.get('performance', {}).get('bias_demand_days', 'N/A')}%",
            "Intervalles de confiance: +/-1.96 * ecart-type historique estime",
        ]

        return XAIResponse(
            product_id=pid, segment=seg,
            prophet_baseline=round(prophet_yhat, 2),
            classifier_probability=pred['probability'],
            threshold=OPTIMAL_THRESHOLD,
            final_prediction=pred['predicted_demand'],
            confidence_interval={
                'low': pred['confidence_low'],
                'high': pred['confidence_high'],
                'method': '+/-1.96 * historical_std_estimate'},
            top_factors=top_factors,
            model_components=model_components,
            assumptions=assumptions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explainability error: {str(e)}")


# ============================================================================
# PREPARATION ORDER
# ============================================================================

@app.post("/preparation-order", response_model=PreparationOrderResponse)
async def generate_preparation_order(request: PreparationOrderRequest):
    """Generate a preparation order based on forecasted demand with optional overrides."""
    try:
        forecast_date = datetime.combine(request.date, datetime.min.time())
        overrides = request.override_quantities or {}

        items = []
        picking_items = []

        for pid in all_product_ids:
            pred = predict_single_product(pid, forecast_date)

            if pid in overrides:
                qty = overrides[pid]
                override_applied = True
            else:
                qty = pred['predicted_demand']
                override_applied = False

            if qty <= 0:
                continue

            stored = warehouse_state.find_product_slots(pid)
            if stored:
                slot_id = stored[0]
                loc = warehouse_state.loc_by_id.get(slot_id, {})
                loc_code = str(loc.get('code_emplacement', 'N/A'))
            else:
                slot_id = 0
                loc_code = "Non stocke"

            items.append(PreparationOrderItem(
                product_id=pid,
                predicted_demand=round(pred['predicted_demand'], 2),
                override_applied=override_applied,
                source_location=loc_code,
                source_location_id=slot_id,
                quantity_to_prepare=round(qty, 2),
                status="ready" if stored else "missing_stock"))

            if stored:
                picking_items.append(PickingItem(
                    product_id=pid,
                    quantity=max(1, int(round(qty))),
                    location_id=slot_id))

        picking_route = None
        if picking_items:
            picking_request = PickingRequest(items=picking_items)
            picking_route = await optimize_picking(picking_request)

        return PreparationOrderResponse(
            date=str(request.date),
            total_items=len(items),
            items=items,
            picking_route=picking_route)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preparation order error: {str(e)}")


# ============================================================================
# MODEL INFO & ASSUMPTIONS
# ============================================================================

@app.get("/model-info")
async def model_info():
    """Full model documentation, assumptions, and performance metrics."""
    return {
        "model_version": "v9 - Prophet + XGBoost Classifier Blend",
        "training_data": config.get("data_info", {}),
        "performance": config.get("performance", {}),
        "architecture": {
            "forecasting": {
                "prophet": "Per-SKU Prophet: cps=0.01, sps=0.1, multiplicative, "
                           "5 temporal regressors (day_of_week, is_weekend, month, is_month_start, is_month_end)",
                "classifier": "XGBoost binary (demand yes/no) with 123 features",
                "blend": f"alpha={ENSEMBLE_ALPHA} * Prophet + (1-alpha)*EWM, bias_mult={BIAS_MULTIPLIER}",
                "calibration": "Per-product calibration factor"
            },
            "storage": {
                "method": "Demand-frequency-aware: HF->PICKING, LF->RESERVE",
                "scoring": "Distance expedition + floor penalty (heavy -> ground)",
                "state": "In-memory occupancy tracking per slot"
            },
            "picking": {
                "method": "Nearest-neighbor heuristic, Manhattan 3D",
                "chariots": f"Multi-chariot (capacity {CHARIOT_CAPACITY_KG}kg)",
                "congestion": "Zone visit counting with warnings"
            }
        },
        "assumptions": [
            "Entrepot B7 avec zones PICKING (RDC) et RESERVE (N1-N4, sous-sol)",
            "HF (233 produits) priorite PICKING, LF (896 produits) en RESERVE",
            "Un emplacement = un produit (pas de multi-reference)",
            f"Chariot: {CHARIOT_CAPACITY_KG}kg max",
            "Distance Manhattan 3D pour routes",
            "Calibration Prophet sur 30 derniers jours entrainement",
            "Intervalles confiance: +/-1.96 * sigma estime",
            "Donnees mars 2024 a janvier 2026"
        ],
        "limitations": [
            f"WAPE={config.get('performance', {}).get('wape_demand_days', 'N/A')}% (objectif <50%)",
            "90.3% sparsite dans les donnees",
            "Features lag/rolling approximes a l'inference",
            "Coordonnees RESERVE a 0 (pas de layout detaille)"
        ]
    }


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
