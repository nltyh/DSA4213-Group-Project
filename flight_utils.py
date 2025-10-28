# flights_utils.py
import re
from typing import Optional, Tuple, List
import pandas as pd

_FLIGHT_ROW_RE = re.compile(
    r"""Outbound:\s*
        (?P<origin>[A-Z]{3})\s+
        (?P<out_dep_dt>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\s*→\s*
        (?P<destination>[A-Z]{3})\s+
        (?P<out_arr_dt>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\s*\|\s*
        (?P<airlines>[^|]+?)\s*\|\s*
        (?P<currency>[A-Z]{3})\s*
        (?P<price>[\d,\.]+)\s*\|\s*
        A(?P<adults>\d+)\s*C(?P<children>\d+)\s*
        (?P<cabin>[A-Z ]+?)\s*\|\|\s*
        Return\s*:\s*
        (?P<ret_origin>[A-Z]{3})\s+
        (?P<ret_dep_dt>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\s*→\s*
        (?P<ret_destination>[A-Z]{3})\s+
        (?P<ret_arr_dt>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})
    """,
    re.VERBOSE
)

def parse_flights_df(df: pd.DataFrame, summary_col: str = "summary") -> pd.DataFrame:
    """
    Parse a one-column summary DataFrame into a structured flights table.
    Returns a new DataFrame with typed columns and derived durations/prices.
    """
    if summary_col not in df.columns:
        raise KeyError(f"Column '{summary_col}' not found in df")

    parsed = df[summary_col].str.extract(_FLIGHT_ROW_RE)

    parsed["adults"] = parsed["adults"].fillna(0).astype(int)
    parsed["children"] = parsed["children"].fillna(0).astype(int)

    for c in ["out_dep_dt", "out_arr_dt", "ret_dep_dt", "ret_arr_dt"]:
        parsed[c] = pd.to_datetime(parsed[c], errors="coerce")

    parsed["price"] = (
        parsed["price"].astype(str).str.replace(",", "", regex=False).astype(float)
    )

    parsed["cabin"] = parsed["cabin"].astype(str).str.strip()
    parsed["airlines"] = parsed["airlines"].astype(str).str.strip()
    parsed["currency"] = parsed["currency"].astype(str).str.strip()

    
    parsed["out_dur_h"] = (parsed["out_arr_dt"] - parsed["out_dep_dt"]).dt.total_seconds() / 3600
    parsed["ret_dur_h"] = (parsed["ret_arr_dt"] - parsed["ret_dep_dt"]).dt.total_seconds() / 3600
    parsed["total_dur_h"] = parsed["out_dur_h"].fillna(0) + parsed["ret_dur_h"].fillna(0)

    parsed["total_price"] = parsed["price"]

    return parsed

_WINDOWS = {
    "morning":  (5, 12),
    "afternoon":(12, 17),
    "evening":  (17, 22),
    "night":    (22, 24),  
}

def _in_window(ts: pd.Timestamp, win: Optional[str]) -> bool:
    if win is None or pd.isna(ts):
        return True
    s, e = _WINDOWS[win]
    h = ts.hour
    return (s <= h < e) or (win == "night" and h < 5)

def pick_cheapest(
    table: pd.DataFrame,
    *,
    origin: str,
    destination: str,
    start_date,  # str | date | Timestamp
    end_date,    # str | date | Timestamp
    fare_type: str = "ECONOMY",
    depart_window: Optional[str] = None,
    arrive_window: Optional[str] = None,
    return_window: Optional[str] = None,
    return_top_n: int = 1,
) -> Tuple[Optional[pd.Series], pd.DataFrame]:
    """
    Deterministically pick the cheapest valid itinerary.
    Returns (best_row, candidates_sorted).
    """
    sd = pd.to_datetime(start_date).date()
    ed = pd.to_datetime(end_date).date()

    cand = table[
        (table["origin"] == origin) &
        (table["destination"] == destination) &
        (table["cabin"].str.upper() == fare_type.upper()) &
        (table["out_dep_dt"].dt.date == sd)
    ].copy()

    if "ret_dep_dt" in cand:
        cand = cand[cand["ret_dep_dt"].dt.date == ed]

    if depart_window:
        cand = cand[cand["out_dep_dt"].apply(lambda t: _in_window(t, depart_window))]
    if arrive_window:
        cand = cand[cand["out_arr_dt"].apply(lambda t: _in_window(t, arrive_window))]
    if return_window and "ret_dep_dt" in cand:
        cand = cand[cand["ret_dep_dt"].apply(lambda t: _in_window(t, return_window))]

    if cand.empty:
        return None, cand

    sort_cols, ascending = ["total_price"], [True]
    if "total_dur_h" in cand.columns:
        sort_cols += ["total_dur_h"]; ascending += [True]

    cand = cand.sort_values(sort_cols, ascending=ascending)
    best = cand.iloc[0] if not cand.empty else None
    return (best, cand.head(return_top_n) if return_top_n > 1 else cand)

def format_flight_row(r: Optional[pd.Series]) -> str:
    if r is None:
        return "No matching flights."
    return (
        f"{r['airlines']} | {r['origin']} {r['out_dep_dt']} → {r['destination']} {r['out_arr_dt']} "
        f"|| Return {r['ret_origin']} {r['ret_dep_dt']} → {r['ret_destination']} {r['ret_arr_dt']} "
        f"| {r['cabin']} | {r['currency']} {r['total_price']:.2f} | ~{r['total_dur_h']:.1f}h"
    )
