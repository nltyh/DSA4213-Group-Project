from amadeus import Client, ResponseError, NetworkError
import json,re, http.client, ssl, certifi
import pandas as pd
from typing import Optional, Tuple
from urllib.parse import urlencode
from datetime import date
from typing import Optional, Literal
from pydantic import BaseModel

# --- Flight Search API ---
TimeWindow = Literal["morning","afternoon","evening","night"]

class TripQuery(BaseModel):
    origin: str
    destination: str
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    adults: int = 1
    children: int = 0
    cabin: Literal["ECONOMY","PREMIUM_ECONOMY","BUSINESS","FIRST"] = "ECONOMY"
    depart_window: Optional[TimeWindow] = None
    arrive_window: Optional[TimeWindow] = None
    return_window: Optional[TimeWindow] = None
    hotel_country: Optional[str] = None
    hotel_city: Optional[str] = None
    hotel_min_rating: Optional[float] = None
    hotel_prefs_text: Optional[str] = None
    
def search_flights_api(query: TripQuery) -> list:
    #HEADERS = {
        #'x-rapidapi-key': "24a885e2b0msh7f84b128bea0d8fp1ddb72jsn845249d7f278",
        #'x-rapidapi-host': "booking-com15.p.rapidapi.com",
    #}
    # -----------------------
    params = {
        "fromId": f"{query.origin}.AIRPORT",
        "toId": f"{query.destination}.AIRPORT",
        "adults": query.adults,
        "children": query.children,
        "cabinClass": query.cabin,
        # Static params to be optimised
        "stops": "none",
        "pageNo": "1",
        "sort": "BEST",
        "currency_code": "SGD" # can make dynamic if needed
    }
    
    if query.start_date:
        params["departDate"] = query.start_date.isoformat()
        
    if query.end_date:
        params["returnDate"] = query.end_date.isoformat()
    else:
        pass
    
    endpoint_path = "/api/v1/flights/searchFlights"
    query_string = urlencode(params)
    full_endpoint = f"{endpoint_path}?{query_string}"

    print(f"Requesting endpoint: {full_endpoint}")

    conn = http.client.HTTPSConnection("booking-com15.p.rapidapi.com", context=ssl.create_default_context(cafile=certifi.where()))
    
    try:
        # --- Send request ---
        conn.request("GET", full_endpoint, headers=HEADERS)
        res = conn.getresponse()
        
        print(f"API Response Status: {res.status}")

        if res.status != 200:
            print(f"Error: API returned status {res.status}")
            print(res.read().decode("utf-8"))
            return [f"Error: API returned status {res.status}"]

        data = res.read()

        # --- Decode JSON ---
        result = json.loads(data.decode("utf-8"))

        # --- Summarize results ---
        lines = summarize_flights(
            result, 
            limit=5, 
            adults=query.adults, 
            children=query.children, 
            cabinClass=query.cabin
        )
        return lines

    except Exception as e:
        print(f"An error occurred: {e}")
        return [f"An error occurred: {e}"]
    finally:
        conn.close()


# --- summarise flight results ---
def summarize_flights(
    result: dict,
    limit: int = 10,
    print_lines: bool = True,
    *,
    adults: int | None = None,
    children: int | str | list | tuple | None = None,
    cabinClass: str | None = None
):
    def _price_from_breakdown(d):
        u = d.get("unifiedPriceBreakdown", {})
        if isinstance(u, dict) and "price" in u:
            p = u["price"]; return p.get("currencyCode"), p.get("units"), p.get("nanos", 0)
        pb = d.get("priceBreakdown", {})
        if isinstance(pb, dict) and "total" in pb:
            p = pb["total"]; return p.get("currencyCode"), p.get("units"), p.get("nanos", 0)
        return None, None, None

    def _fmt_money(ccy, units, nanos):
        if ccy is None or units is None: return "?"
        amt = float(units) + (float(nanos or 0) / 1_000_000_000.0)
        return f"{ccy} {amt:,.2f}"

    def _carriers_from_segments(segments):
        names = []
        for seg in segments or []:
            for leg in seg.get("legs", []):
                for cd in leg.get("carriersData", []):
                    n = cd.get("name")
                    if n and n not in names: names.append(n)
            for code in seg.get("carriers", []):
                if code and code not in names: names.append(code)
        return names or ["Unknown"]

    def _dep_arr_from_segment(seg):
        if not seg: return None, None, None, None
        dep_air = (seg.get("departureAirport") or {}).get("code")
        dep_time = seg.get("departureTime")
        arr_air = (seg.get("arrivalAirport") or {}).get("code")
        arr_time = seg.get("arrivalTime")
        return dep_air, dep_time, arr_air, arr_time

    def _children_count(val):
        if val is None: return None
        if isinstance(val, int): return val
        if isinstance(val, (list, tuple)): return len(val)
        if isinstance(val, str):
            s = val.strip()
            if not s: return 0
            return len([x for x in s.split(",") if x.strip() != ""])
        return None

    def _infer_fare_type(offer, segments, cabin_arg):
        if cabin_arg: return cabin_arg
        seg0 = segments[0] if segments else None
        if isinstance(seg0, dict):
            v = seg0.get("cabinClass")
            if not v and isinstance(seg0.get("legs"), list) and seg0["legs"]:
                v = seg0["legs"][0].get("cabinClass")
            if v: return v
        return offer.get("cabinClass")

    candidates = []
    if isinstance(result, list):
        candidates = result
    elif isinstance(result, dict):
        data = result.get("data")
        if isinstance(data, list):
            candidates = data
        elif isinstance(data, dict):
            for key in ("itineraries", "offers", "results", "items", "searchResults", "flightResults"):
                if isinstance(data.get(key), list):
                    candidates = data[key]; break
        if not candidates:
            for key in ("itineraries", "offers", "results", "items", "searchResults", "flightResults"):
                if isinstance(result.get(key), list):
                    candidates = result[key]; break

    def _looks_like_offer(x):
        return isinstance(x, dict) and ("segments" in x and isinstance(x.get("segments"), list)) and ("unifiedPriceBreakdown" in x or "priceBreakdown" in x)

    def _deep_find_offers(node, found):
        if isinstance(node, list):
            if node and all(isinstance(el, dict) for el in node):
                offerish = [el for el in node if _looks_like_offer(el)]
                if offerish:
                    found.extend(node)
                    return
            for el in node: _deep_find_offers(el, found)
        elif isinstance(node, dict):
            for v in node.values(): _deep_find_offers(v, found)

    if not candidates:
        found = []
        _deep_find_offers(result, found)
        candidates = found

    offers = [o for o in candidates if _looks_like_offer(o)]
    if not offers:
        return ["No offers found"]

    ch_ct = _children_count(children)

    lines = []
    for i, offer in enumerate(offers[:limit], 1):
        ccy, units, nanos = _price_from_breakdown(offer)
        price_str = _fmt_money(ccy, units, nanos)

        segments = offer.get("segments", [])
        out_seg = segments[0] if len(segments) >= 1 else None
        ret_seg = segments[1] if len(segments) >= 2 else None

        carriers = ", ".join(_carriers_from_segments(segments))

        out_dep, out_dt, out_arr, out_at = _dep_arr_from_segment(out_seg)

        fare = _infer_fare_type(offer, segments, cabinClass)
        extras = []
        if adults is not None: extras.append(f"A{adults}")
        if ch_ct is not None:  extras.append(f"C{ch_ct}")
        if fare:               extras.append(fare)

        extra_txt = f" | {' '.join(extras)}" if extras else ""

        line = f"{i:02d}. Outbound: {out_dep or '?'} {out_dt or '?'} → {out_arr or '?'} {out_at or '?'} | {carriers} | {price_str}{extra_txt}"
        lines.append(line)
        
        if ret_seg is not None:
            ret_dep, ret_dt, ret_arr, ret_at = _dep_arr_from_segment(ret_seg)
            ret_line = f"    Return : {ret_dep or '?'} {ret_dt or '?'} → {ret_arr or '?'} {ret_at or '?'}"
            lines.append(ret_line)

    if print_lines:
        for line in lines:
            print(line)
    return lines


#--- static flight utils --- 
_FLIGHT_ROW_RE = re.compile(
    r"""^\s*\d{0,2}\.?
        \s*Outbound:\s*
        (?P<origin>[A-Z]{3})\s+
        (?P<out_dep_dt>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\s*→\s*
        (?P<destination>[A-Z]{3})\s+
        (?P<out_arr_dt>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\s*\|\s*
        (?P<airlines>[^|]+?)\s*\|\s*
        (?P<currency>[A-Z]{3})\s*
        (?P<price>[\d,\.]+)\s*\|\s*
        A(?P<adults>\d+)\s*C(?P<children>\d+)\s*
        (?P<cabin>[A-Z_ ]+?)\s*\|\|\s*
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

def pick_flight(
    table: pd.DataFrame,
    *,
    origin: str,
    destination: str,
    start_date,  
    end_date,    
    fare_type: str
) -> Tuple[Optional[pd.Series], pd.DataFrame]:
    sd = pd.to_datetime(start_date).date()
    ed = pd.to_datetime(end_date).date()

    cand = table[
        (table["origin"] == origin) &
        (table["destination"] == destination) &
        (table["cabin"].str.upper() == fare_type.upper()) &
        (table["out_dep_dt"].dt.date == sd)
    ].copy()

    return cand

def _dtfmt(ts): 
    return pd.to_datetime(ts).strftime("%d %b %Y %H:%M")

def flights_context(cands: pd.DataFrame)->str:
    if cands.empty:
        return "No flight options matched the filters."
    lines=[]
    for _, r in cands.iterrows():
        lines.append(
            f"* {r['airlines']} — {r['origin']} {_dtfmt(r['out_dep_dt'])} → {r['destination']} {_dtfmt(r['out_arr_dt'])} "
            f"|| Return {r['ret_origin']} {_dtfmt(r['ret_dep_dt'])} → {r['ret_destination']} {_dtfmt(r['ret_arr_dt'])} "
            f"| {r['currency']} {r['total_price']:.2f} | A{r['adults']} C{r['children']} {r['cabin'].upper()}"
        )
    return "\n".join(lines)
