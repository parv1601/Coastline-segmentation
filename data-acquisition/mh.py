import requests
import pandas as pd

LAYER_QUERY_URL = "https://gisportal.ncscm.res.in/server/rest/services/CZMP_STATES/INDIA_CZMPPDF/MapServer/15/query"

def polygon_bbox(rings):
    xs, ys = [], []
    for ring in rings:
        for x, y in ring:
            xs.append(x)
            ys.append(y)
    return min(xs), min(ys), max(xs), max(ys)

def fetch_all_features():
    all_features = []
    offset = 0
    batch_size = 1000

    while True:
        params = {
            "f": "json",
            "where": "1=1",
            "outFields": "OBJECTID,INDEX_NO_1",
            "returnGeometry": "true",
            "outSR": "4326",

            # correct pagination for Maharashtra
            "orderByFields": "OBJECTID ASC",

            # pagination
            "resultOffset": offset,
            "resultRecordCount": batch_size
        }

        r = requests.get(LAYER_QUERY_URL, params=params, timeout=60)
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

def main():
    features = fetch_all_features()
    print("Maharashtra: downloaded", len(features), "features")

    rows = []
    for i, feat in enumerate(features):
        geom = feat.get("geometry", {})
        attrs = feat.get("attributes", {})

        if "rings" not in geom:
            continue

        min_lon, min_lat, max_lon, max_lat = polygon_bbox(geom["rings"])

        rows.append({
            "grid_id": i + 1,
            "min_lon": min_lon,
            "min_lat": min_lat,
            "max_lon": max_lon,
            "max_lat": max_lat,

            # Maharashtra uses INDEX_NO_1
            "INDEX_NO": attrs.get("INDEX_NO_1", None)
        })

    df = pd.DataFrame(rows)
    out_file = "maharashtra_grid_bbox.csv"
    df.to_csv(out_file, index=False)

    print("Saved ->", out_file)

if __name__ == "__main__":
    main()
