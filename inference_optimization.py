"""
MobAI'26 - Task 1: Warehouse Optimization - Inference Script
=============================================================
Standalone script that:
  1. Loads warehouse data and AI models
  2. Reads test data CSV (Date, Product, Flow Type, Quantity)
  3. Simulates warehouse operations chronologically
  4. Exports operational instructions CSV:
       Product, Action, Location, Route, Reason

Usage:
  python inference_optimization.py --input test_data.csv --output optimization_results.csv
  python inference_optimization.py --input test_data.csv  # default output name

Output format matches submission guide:
  Product,Action,Location,Route,Reason
  31554,Storage,0H-01-02,Reception->B7->Picking,HF product - min distance
  31565,Picking,0X-05-03,Picking->Expedition,High demand - priority route
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

CHARIOT_CAPACITY_KG = 300.0
RECEIPT_ZONE = {'x': 0, 'y': 0, 'z': 0}
EXPEDITION_ZONE = {'x': 3, 'y': 5, 'z': 0}


# ============================================================================
# MODEL & DATA LOADING
# ============================================================================

def load_all():
    """Load models, metadata, and warehouse data."""
    print("[1/4] Loading models and warehouse data...")
    t0 = time.time()

    # XGBoost classifier
    classifier = xgb.Booster()
    classifier.load_model(str(MODELS_DIR / "xgboost_classifier_model.json"))

    # Config
    with open(MODELS_DIR / "forecast_config.json", 'r') as f:
        config = json.load(f)

    # Prophet metadata
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
    warehouse_locations = pd.read_csv(DATA_DIR / "warehouse_locations.csv")

    seg_lookup = dict(zip(product_segments["id_produit"], product_segments["segment"]))
    priority_lookup = product_priorities.set_index("id_produit").to_dict("index")

    elapsed = time.time() - t0
    print(f"   Loaded in {elapsed:.1f}s")
    print(f"   Products: {len(product_priorities)}, Locations: {len(warehouse_locations)}")

    return {
        'classifier': classifier,
        'config': config,
        'prophet_meta': prophet_meta,
        'product_attrs': product_attrs,
        'delivery_stats': delivery_stats,
        'cat_encoding': cat_encoding,
        'seg_lookup': seg_lookup,
        'priority_lookup': priority_lookup,
        'warehouse_locations': warehouse_locations,
    }


# ============================================================================
# WAREHOUSE STATE
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
        return {
            'total_slots': total,
            'occupied_slots': occupied,
            'available_slots': total - occupied,
            'occupancy_rate': round(occupied / total * 100, 1) if total > 0 else 0,
        }


# ============================================================================
# STORAGE ASSIGNMENT
# ============================================================================

def assign_storage(product_id, quantity, state, data, weight_kg=None):
    """Demand-frequency-aware storage assignment: HF -> PICKING, LF -> RESERVE."""
    seg_lookup = data['seg_lookup']
    priority_lookup = data['priority_lookup']

    pp = priority_lookup.get(product_id, {})
    priority_score = pp.get("priority_score", 0.0)
    p_freq = pp.get("demand_frequency", 0.0)
    seg = seg_lookup.get(product_id, "LF")

    # HF products go to PICKING (close to expedition), LF to RESERVE
    if seg == "HF" and p_freq > 0.01:
        slot_type = "PICKING"
    else:
        slot_type = "RESERVE"

    available = state.get_available_slots(slot_type)
    if len(available) == 0:
        # Fallback to other type
        slot_type = "PICKING" if slot_type == "RESERVE" else "RESERVE"
        available = state.get_available_slots(slot_type)

    if len(available) == 0:
        return None  # No slots available

    available = available.copy()
    available['dist_exp'] = pd.to_numeric(available['dist_from_expedition'], errors='coerce').fillna(8)
    z_vals = pd.to_numeric(available['z'], errors='coerce').fillna(0)

    # Score: prefer close to expedition, prefer ground floor
    if seg == "HF":
        available['score'] = -available['dist_exp'] * 3.0 - z_vals * 5.0
    else:
        available['score'] = -available['dist_exp'] * 1.0 - z_vals * 2.0

    # Heavy items get ground floor preference
    prod_weight = weight_kg or pp.get("weight", 0.0)
    if prod_weight and prod_weight > 5:
        available['score'] = available['score'] - z_vals * 10.0

    best = available.nlargest(1, 'score').iloc[0]
    slot_id = int(best['id_emplacement'])
    state.assign_slot(slot_id, product_id, quantity)

    loc_code = str(best['code_emplacement'])
    zone = str(best.get('zone', ''))

    # Build route description
    route = build_route(zone, loc_code, "ingoing")

    # Build reason
    reason_parts = [f"Segment={seg}"]
    if seg == "HF":
        reason_parts.append(f"High freq ({p_freq:.3f})")
        reason_parts.append("Min distance to expedition")
    else:
        reason_parts.append(f"Low freq ({p_freq:.4f})")
        reason_parts.append("Reserve storage")

    return {
        'location_code': loc_code,
        'location_id': slot_id,
        'zone': zone,
        'route': route,
        'reason': '; '.join(reason_parts),
        'action': 'Storage',
    }


def build_route(zone, loc_code, flow_type):
    """Build human-readable route description."""
    if flow_type == "ingoing":
        if 'PCK' in zone:
            return f"Reception -> Zone Picking B7 -> {loc_code}"
        elif 'B07-N' in zone:
            level = zone.replace('B07-', '')
            return f"Reception -> Lift -> {level} -> {loc_code}"
        elif 'B07-SS' in zone:
            return f"Reception -> Lift -> Sous-sol -> {loc_code}"
        else:
            return f"Reception -> {zone} -> {loc_code}"
    else:  # outgoing
        if 'PCK' in zone:
            return f"{loc_code} -> Zone Picking B7 -> Expedition"
        elif 'B07-N' in zone:
            level = zone.replace('B07-', '')
            return f"{loc_code} -> {level} -> Lift -> Expedition"
        elif 'B07-SS' in zone:
            return f"{loc_code} -> Sous-sol -> Lift -> Expedition"
        else:
            return f"{loc_code} -> {zone} -> Expedition"


# ============================================================================
# PICKING (OUTGOING)
# ============================================================================

def process_outgoing(product_id, quantity, state, data):
    """Pick product from stored location and route to expedition."""
    seg_lookup = data['seg_lookup']
    priority_lookup = data['priority_lookup']

    stored_slots = state.find_product_slots(product_id)
    if not stored_slots:
        return {
            'location_code': 'N/A',
            'zone': '',
            'route': 'N/A',
            'reason': f'Product {product_id} not in stock - stockout',
            'action': 'Picking',
            'success': False,
        }

    slot_id = stored_slots[0]
    slot_info = state.loc_by_id.get(slot_id, {})
    loc_code = str(slot_info.get('code_emplacement', 'UNKNOWN'))
    zone = str(slot_info.get('zone', ''))
    seg = seg_lookup.get(product_id, "LF")

    route = build_route(zone, loc_code, "outgoing")

    pp = priority_lookup.get(product_id, {})
    p_freq = pp.get("demand_frequency", 0.0)

    reason_parts = []
    if seg == "HF":
        reason_parts.append("High demand product")
    else:
        reason_parts.append("Low frequency product")
    reason_parts.append(f"Freq={p_freq:.4f}")

    # Release the slot
    state.release_slot(slot_id)

    return {
        'location_code': loc_code,
        'zone': zone,
        'route': route,
        'reason': '; '.join(reason_parts),
        'action': 'Picking',
        'success': True,
    }


# ============================================================================
# MAIN INFERENCE
# ============================================================================

def run_inference(input_file, output_file, data):
    """Process test events and generate operational instructions CSV."""
    print(f"[2/4] Reading test data from {input_file}...")

    # Read input CSV - flexible column detection
    df = pd.read_csv(input_file)
    print(f"   Input shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")

    # Normalize column names (handle various formats)
    col_map = {}
    for c in df.columns:
        cl = c.strip().lower().replace(' ', '_')
        if 'date' in cl:
            col_map[c] = 'date'
        elif 'product' in cl or 'produit' in cl:
            col_map[c] = 'product_id'
        elif 'flow' in cl or 'type' in cl:
            col_map[c] = 'flow_type'
        elif 'quantit' in cl or 'qty' in cl:
            col_map[c] = 'quantity'
    df = df.rename(columns=col_map)

    required = ['date', 'product_id', 'flow_type', 'quantity']
    for col in required:
        if col not in df.columns:
            print(f"   ERROR: Missing column '{col}' in input. Found: {list(df.columns)}")
            sys.exit(1)

    # Parse product_id as int
    df['product_id'] = pd.to_numeric(df['product_id'], errors='coerce').fillna(0).astype(int)
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0).astype(int)

    # Sort by date for chronological processing
    try:
        df['date_parsed'] = pd.to_datetime(df['date'], dayfirst=True)
    except Exception:
        df['date_parsed'] = pd.to_datetime(df['date'])
    df = df.sort_values('date_parsed').reset_index(drop=True)

    print(f"   Events: {len(df)}")
    print(f"   Date range: {df['date'].iloc[0]} -> {df['date'].iloc[-1]}")
    print(f"   Flow types: {df['flow_type'].value_counts().to_dict()}")

    # Initialize warehouse state
    print("[3/4] Processing events...")
    wh_state = WarehouseState(data['warehouse_locations'])
    results = []

    for idx, row in df.iterrows():
        pid = int(row['product_id'])
        qty = int(row['quantity'])
        flow = str(row['flow_type']).strip().lower()

        if flow in ['ingoing', 'in', 'entree', 'reception', 'incoming']:
            result = assign_storage(pid, qty, wh_state, data)
            if result:
                results.append({
                    'Product': pid,
                    'Action': result['action'],
                    'Location': result['location_code'],
                    'Route': result['route'],
                    'Reason': result['reason'],
                })
            else:
                results.append({
                    'Product': pid,
                    'Action': 'Storage',
                    'Location': 'N/A',
                    'Route': 'N/A',
                    'Reason': 'No available slots',
                })
        elif flow in ['outgoing', 'out', 'sortie', 'expedition', 'outcoming']:
            result = process_outgoing(pid, qty, wh_state, data)
            results.append({
                'Product': pid,
                'Action': result['action'],
                'Location': result['location_code'],
                'Route': result['route'],
                'Reason': result['reason'],
            })
        else:
            results.append({
                'Product': pid,
                'Action': 'Unknown',
                'Location': 'N/A',
                'Route': 'N/A',
                'Reason': f'Unknown flow type: {flow}',
            })

        if (idx + 1) % 100 == 0:
            print(f"   Processed {idx+1}/{len(df)} events...", end='\r')

    print(f"   Processed {len(df)} events total         ")

    # Write output
    print(f"[4/4] Writing output to {output_file}...")
    df_out = pd.DataFrame(results)
    df_out.to_csv(output_file, index=False)

    # Summary
    occupancy = wh_state.get_occupancy_stats()
    print(f"\n{'='*60}")
    print(f"  WAREHOUSE OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Output file:       {output_file}")
    print(f"  Total events:      {len(results)}")
    print(f"  Storage actions:   {sum(1 for r in results if r['Action'] == 'Storage')}")
    print(f"  Picking actions:   {sum(1 for r in results if r['Action'] == 'Picking')}")
    print(f"  Failures:          {sum(1 for r in results if 'N/A' in r['Location'])}")
    print(f"  Final occupancy:   {occupancy['occupied_slots']}/{occupancy['total_slots']} "
          f"({occupancy['occupancy_rate']}%)")
    print(f"{'='*60}")

    # Show sample output
    print("\n  Sample output (first 5 rows):")
    for _, row in df_out.head(5).iterrows():
        print(f"    {row['Product']}, {row['Action']}, {row['Location']}, "
              f"{row['Route'][:40]}..., {row['Reason'][:40]}...")

    return df_out


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MobAI'26 Task 1 - Warehouse Optimization Inference")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to test data CSV (Date, Product, Flow Type, Quantity)")
    parser.add_argument("--output", type=str, default="optimization_results.csv",
                        help="Path for output CSV (default: optimization_results.csv)")
    args = parser.parse_args()

    print("=" * 60)
    print("  MobAI'26 - Task 1: Warehouse Optimization Inference")
    print("=" * 60)

    # Load models and data
    data = load_all()

    # Run inference
    df_result = run_inference(args.input, args.output, data)

    print(f"\nDone! Output saved to: {args.output}")
