import requests
import pandas as pd

STATE_LAYERS = {
    "maharashtra": "https://gisportal.ncscm.res.in/server/rest/services/CZMP_STATES/INDIA_CZMPPDF/MapServer/15/query",
    "goa":         "https://gisportal.ncscm.res.in/server/rest/services/CZMP_STATES/INDIA_CZMPPDF/MapServer/16/query",
    "karnataka":   "https://gisportal.ncscm.res.in/server/rest/services/CZMP_STATES/INDIA_CZMPPDF/MapServer/17/query",
    "kerala":      "https://gisportal.ncscm.res.in/server/rest/services/CZMP_STATES/INDIA_CZMPPDF/MapServer/18/query",
}

def polygon_bbox(rings):
    xs, ys = [], []
    for ring in rings:
        for x, y in ring:
            xs.append(x)
            ys.append(y)
    return min(xs), min(ys), max(xs), max(ys)

def fetch_all_features(layer_query_url):
    all_features = []
    offset = 0
    batch_size = 1000

    while True:
        params = {
            "f": "json",
            "where": "1=1",
            "outFields": "INDEX_NO,OBJECTID_1,OBJECTID",
            "returnGeometry": "true",
            "outSR": "4326",

            # IMPORTANT FIX for stable pagination:
            "orderByFields": "OBJECTID_1 ASC",

            # pagination
            "resultOffset": offset,
            "resultRecordCount": batch_size
        }

        r = requests.get(layer_query_url, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()

        feats = data.get("features", [])
        if not feats:
            break

        all_features.extend(feats)

        if len(feats) < batch_size:
            break

        offset += batch_size

    return all_features

def export_state_csv(state_name, layer_query_url):
    features = fetch_all_features(layer_query_url)
    print(f"{state_name}: downloaded {len(features)} features")

    rows = []
    for i, feat in enumerate(features):
        geom = feat.get("geometry", {})
        attrs = feat.get("attributes", {})

        if "rings" not in geom:
            continue

        min_lon, min_lat, max_lon, max_lat = polygon_bbox(geom["rings"])

        rows.append({
            "grid_id": i + 1,   # indexing from 1
            "min_lon": min_lon,
            "min_lat": min_lat,
            "max_lon": max_lon,
            "max_lat": max_lat,
            "INDEX_NO": attrs.get("INDEX_NO", None)
        })

    df = pd.DataFrame(rows)
    out_file = f"{state_name}_grid_bbox.csv"
    df.to_csv(out_file, index=False)

    print(f"Saved -> {out_file}")
    return df

def main():
    for state_name, url in STATE_LAYERS.items():
        try:
            export_state_csv(state_name, url)
        except Exception as e:
            print(f"Failed for {state_name}: {e}")

if __name__ == "__main__":
    main()
