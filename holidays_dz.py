"""
holidays_dz.py — Algerian & Islamic holiday calendar for demand forecasting.

Provides holiday features that significantly impact warehouse demand patterns:
- Ramadan (30 days of altered consumption)
- Eid al-Fitr & Eid al-Adha (major shopping spikes before, drops during)
- Algerian national holidays (fixed dates)
- Mawlid, Amazigh New Year, etc.

Islamic dates are approximated from the Hijri calendar.
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta

# ============================================================================
# ISLAMIC HOLIDAYS (approximate Gregorian dates)
# The Hijri calendar shifts ~11 days earlier each Gregorian year.
# These are best-known approximations; actual dates depend on moon sighting.
# ============================================================================

# Ramadan start dates (1st Ramadan)
RAMADAN_START = {
    2023: date(2023, 3, 23),
    2024: date(2024, 3, 11),
    2025: date(2025, 3, 1),
    2026: date(2026, 2, 18),
    2027: date(2027, 2, 8),
    2028: date(2028, 1, 28),
}

# Eid al-Fitr (1 Shawwal) — end of Ramadan, 2-3 day holiday
EID_AL_FITR = {
    2023: date(2023, 4, 21),
    2024: date(2024, 4, 10),
    2025: date(2025, 3, 30),
    2026: date(2026, 3, 20),
    2027: date(2027, 3, 10),
    2028: date(2028, 2, 27),
}

# Eid al-Adha (10 Dhul Hijjah) — 2-3 day holiday
EID_AL_ADHA = {
    2023: date(2023, 6, 28),
    2024: date(2024, 6, 16),
    2025: date(2025, 6, 6),
    2026: date(2026, 5, 27),
    2027: date(2027, 5, 16),
    2028: date(2028, 5, 4),
}

# Mawlid an-Nabi (12 Rabi al-Awwal)
MAWLID = {
    2023: date(2023, 9, 27),
    2024: date(2024, 9, 16),
    2025: date(2025, 9, 5),
    2026: date(2026, 8, 26),
    2027: date(2027, 8, 15),
    2028: date(2028, 8, 3),
}

# Islamic New Year (1 Muharram)
ISLAMIC_NEW_YEAR = {
    2023: date(2023, 7, 19),
    2024: date(2024, 7, 8),
    2025: date(2025, 6, 27),
    2026: date(2026, 6, 17),
    2027: date(2027, 6, 6),
    2028: date(2028, 5, 25),
}

# Ashura (10 Muharram)
ASHURA = {
    2023: date(2023, 7, 28),
    2024: date(2024, 7, 17),
    2025: date(2025, 7, 6),
    2026: date(2026, 6, 26),
    2027: date(2027, 6, 15),
    2028: date(2028, 6, 3),
}

# ============================================================================
# ALGERIAN FIXED NATIONAL HOLIDAYS
# ============================================================================

FIXED_HOLIDAYS = [
    (1,  1,  "new_year"),           # New Year's Day
    (1,  12, "yennayer"),           # Amazigh New Year (Yennayer)
    (5,  1,  "labour_day"),         # Labour Day
    (7,  5,  "independence_day"),   # Independence Day
    (11, 1,  "revolution_day"),     # Revolution Day
]


def _nearest_date(target: date, date_dict: dict) -> tuple:
    """Find nearest occurrence (past or future) from a yearly date dict."""
    best_dist = 99999
    best_d = None
    for yr, d in date_dict.items():
        dist = abs((target - d).days)
        if dist < best_dist:
            best_dist = dist
            best_d = d
    return best_d, best_dist


def get_holiday_features_for_date(d) -> dict:
    """
    Compute all holiday-related features for a single date.
    Returns a dict of float features.
    Accepts both datetime.date and datetime.datetime objects.
    """
    # Normalize to date (handles datetime inputs)
    if hasattr(d, 'date') and callable(d.date):
        d = d.date()
    year = d.year
    features = {}

    # --- Ramadan features ---
    ram_start = RAMADAN_START.get(year)
    ram_start_prev = RAMADAN_START.get(year - 1)
    ram_start_next = RAMADAN_START.get(year + 1)

    # Determine current/nearest Ramadan
    is_ramadan = 0.0
    ramadan_day = 0.0  # day within Ramadan (1-30)
    days_to_ramadan = 999.0
    days_since_ramadan = 999.0
    ramadan_prep = 0.0  # 1-2 weeks before Ramadan (stocking up)

    if ram_start:
        ram_end = ram_start + timedelta(days=29)
        if ram_start <= d <= ram_end:
            is_ramadan = 1.0
            ramadan_day = float((d - ram_start).days + 1)
        elif d < ram_start:
            days_to_ramadan = float((ram_start - d).days)
        else:
            days_since_ramadan = float((d - ram_end).days)

    if ram_start_prev and days_since_ramadan == 999.0:
        ram_end_prev = ram_start_prev + timedelta(days=29)
        if d > ram_end_prev:
            days_since_ramadan = float((d - ram_end_prev).days)

    if ram_start_next and days_to_ramadan == 999.0:
        if d < ram_start_next:
            days_to_ramadan = float((ram_start_next - d).days)

    # Ramadan preparation (14 days before)
    if 0 < days_to_ramadan <= 14:
        ramadan_prep = 1.0

    # Ramadan period features
    features["is_ramadan"] = is_ramadan
    features["ramadan_day"] = ramadan_day
    features["ramadan_progress"] = ramadan_day / 30.0  # 0.0 to 1.0
    features["days_to_ramadan"] = min(days_to_ramadan, 60.0)
    features["days_since_ramadan"] = min(days_since_ramadan, 60.0)
    features["ramadan_prep"] = ramadan_prep
    features["ramadan_last_week"] = 1.0 if is_ramadan and ramadan_day > 23 else 0.0

    # --- Eid al-Fitr features (end of Ramadan, big celebration) ---
    eid_f, eid_f_dist = _nearest_date(d, EID_AL_FITR)
    is_eid_fitr = 0.0
    days_to_eid_fitr = 999.0
    eid_fitr_prep = 0.0

    if eid_f:
        eid_f_period = [eid_f + timedelta(days=i) for i in range(3)]  # 3-day holiday
        if d in eid_f_period:
            is_eid_fitr = 1.0
        diff = (eid_f - d).days
        if diff > 0:
            days_to_eid_fitr = float(diff)
            if diff <= 7:
                eid_fitr_prep = 1.0  # Shopping spike before Eid
        else:
            days_to_eid_fitr = 0.0

    features["is_eid_fitr"] = is_eid_fitr
    features["days_to_eid_fitr"] = min(days_to_eid_fitr, 30.0)
    features["eid_fitr_prep"] = eid_fitr_prep

    # --- Eid al-Adha features (sacrifice feast, major meat/food demand) ---
    eid_a, eid_a_dist = _nearest_date(d, EID_AL_ADHA)
    is_eid_adha = 0.0
    days_to_eid_adha = 999.0
    eid_adha_prep = 0.0

    if eid_a:
        eid_a_period = [eid_a + timedelta(days=i) for i in range(3)]
        if d in eid_a_period:
            is_eid_adha = 1.0
        diff = (eid_a - d).days
        if diff > 0:
            days_to_eid_adha = float(diff)
            if diff <= 10:
                eid_adha_prep = 1.0
        else:
            days_to_eid_adha = 0.0

    features["is_eid_adha"] = is_eid_adha
    features["days_to_eid_adha"] = min(days_to_eid_adha, 30.0)
    features["eid_adha_prep"] = eid_adha_prep

    # --- Other Islamic holidays ---
    mawl, mawl_dist = _nearest_date(d, MAWLID)
    features["is_mawlid"] = 1.0 if mawl and d == mawl else 0.0

    iny, iny_dist = _nearest_date(d, ISLAMIC_NEW_YEAR)
    features["is_islamic_new_year"] = 1.0 if iny and d == iny else 0.0

    ashu, ashu_dist = _nearest_date(d, ASHURA)
    features["is_ashura"] = 1.0 if ashu and (d == ashu or d == ashu + timedelta(days=1)) else 0.0

    # --- Algerian fixed holidays ---
    is_national_holiday = 0.0
    days_to_holiday = 999.0
    for m, dy, name in FIXED_HOLIDAYS:
        hol = date(year, m, dy)
        if d == hol:
            is_national_holiday = 1.0
        diff = (hol - d).days
        if 0 < diff < days_to_holiday:
            days_to_holiday = float(diff)
        # Check next year's Jan holidays too
        if m <= 2:
            hol_next = date(year + 1, m, dy)
            diff2 = (hol_next - d).days
            if 0 < diff2 < days_to_holiday:
                days_to_holiday = float(diff2)

    features["is_national_holiday"] = is_national_holiday
    features["days_to_national_holiday"] = min(days_to_holiday, 30.0)

    # --- Combined holiday indicator ---
    features["is_any_holiday"] = max(
        is_national_holiday, is_eid_fitr, is_eid_adha,
        features["is_mawlid"], features["is_islamic_new_year"]
    )

    # --- Pre-holiday shopping effect (any major holiday within 3 days) ---
    features["pre_holiday_3d"] = 1.0 if (
        (0 < days_to_eid_fitr <= 3) or
        (0 < days_to_eid_adha <= 3) or
        (0 < days_to_holiday <= 3)
    ) else 0.0

    return features


# List of all holiday feature names (for feature column management)
HOLIDAY_FEATURE_NAMES = [
    "is_ramadan", "ramadan_day", "ramadan_progress",
    "days_to_ramadan", "days_since_ramadan", "ramadan_prep", "ramadan_last_week",
    "is_eid_fitr", "days_to_eid_fitr", "eid_fitr_prep",
    "is_eid_adha", "days_to_eid_adha", "eid_adha_prep",
    "is_mawlid", "is_islamic_new_year", "is_ashura",
    "is_national_holiday", "days_to_national_holiday",
    "is_any_holiday", "pre_holiday_3d",
]


def build_prophet_holidays(year_start: int = 2023, year_end: int = 2028) -> pd.DataFrame:
    """
    Build a Prophet-compatible holidays DataFrame.
    Prophet expects columns: holiday, ds, lower_window, upper_window
    """
    rows = []

    for year in range(year_start, year_end + 1):
        # Ramadan (30 days)
        if year in RAMADAN_START:
            rs = RAMADAN_START[year]
            for i in range(30):
                rows.append({
                    "holiday": "ramadan",
                    "ds": pd.Timestamp(rs + timedelta(days=i)),
                    "lower_window": 0,
                    "upper_window": 0,
                })
            # Pre-Ramadan stocking (7 days before)
            for i in range(1, 8):
                rows.append({
                    "holiday": "pre_ramadan",
                    "ds": pd.Timestamp(rs - timedelta(days=i)),
                    "lower_window": 0,
                    "upper_window": 0,
                })

        # Eid al-Fitr (3 days)
        if year in EID_AL_FITR:
            ef = EID_AL_FITR[year]
            for i in range(3):
                rows.append({
                    "holiday": "eid_al_fitr",
                    "ds": pd.Timestamp(ef + timedelta(days=i)),
                    "lower_window": 0,
                    "upper_window": 0,
                })
            # Pre-Eid shopping (5 days before)
            for i in range(1, 6):
                rows.append({
                    "holiday": "pre_eid_fitr",
                    "ds": pd.Timestamp(ef - timedelta(days=i)),
                    "lower_window": 0,
                    "upper_window": 0,
                })

        # Eid al-Adha (3 days)
        if year in EID_AL_ADHA:
            ea = EID_AL_ADHA[year]
            for i in range(3):
                rows.append({
                    "holiday": "eid_al_adha",
                    "ds": pd.Timestamp(ea + timedelta(days=i)),
                    "lower_window": 0,
                    "upper_window": 0,
                })
            for i in range(1, 6):
                rows.append({
                    "holiday": "pre_eid_adha",
                    "ds": pd.Timestamp(ea - timedelta(days=i)),
                    "lower_window": 0,
                    "upper_window": 0,
                })

        # Mawlid
        if year in MAWLID:
            rows.append({
                "holiday": "mawlid",
                "ds": pd.Timestamp(MAWLID[year]),
                "lower_window": 0,
                "upper_window": 0,
            })

        # Islamic New Year
        if year in ISLAMIC_NEW_YEAR:
            rows.append({
                "holiday": "islamic_new_year",
                "ds": pd.Timestamp(ISLAMIC_NEW_YEAR[year]),
                "lower_window": 0,
                "upper_window": 0,
            })

        # Ashura (2 days)
        if year in ASHURA:
            for i in range(2):
                rows.append({
                    "holiday": "ashura",
                    "ds": pd.Timestamp(ASHURA[year] + timedelta(days=i)),
                    "lower_window": 0,
                    "upper_window": 0,
                })

        # Fixed Algerian holidays
        for m, dy, name in FIXED_HOLIDAYS:
            rows.append({
                "holiday": name,
                "ds": pd.Timestamp(date(year, m, dy)),
                "lower_window": 0,
                "upper_window": 0,
            })

    return pd.DataFrame(rows)


def add_holiday_columns_to_df(df: pd.DataFrame, date_col: str = "day") -> pd.DataFrame:
    """
    Add all holiday feature columns to a DataFrame with a date column.
    Vectorized for speed.
    """
    dates = pd.to_datetime(df[date_col])
    n = len(df)

    # Pre-compute all features
    feat_arrays = {name: np.zeros(n) for name in HOLIDAY_FEATURE_NAMES}

    for idx in range(n):
        d = dates.iloc[idx].date()
        feats = get_holiday_features_for_date(d)
        for name, val in feats.items():
            feat_arrays[name][idx] = val

    for name, arr in feat_arrays.items():
        df[name] = arr

    return df


def add_holiday_columns_fast(df: pd.DataFrame, date_col: str = "day") -> pd.DataFrame:
    """
    Optimized version: compute holiday features per unique date, then merge.
    Much faster when many products share the same dates (grid data).
    """
    dates = pd.to_datetime(df[date_col])
    unique_dates = dates.dt.date.unique()

    date_features = {}
    for d in unique_dates:
        date_features[d] = get_holiday_features_for_date(d)

    for name in HOLIDAY_FEATURE_NAMES:
        df[name] = dates.dt.date.map(lambda d, n=name: date_features.get(d, {}).get(n, 0.0))

    return df
