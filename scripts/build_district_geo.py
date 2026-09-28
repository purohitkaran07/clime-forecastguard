"""
Filters the open India-districts GeoJSON (udit-001/india-maps-data, 2011 census
district boundaries) down to the 18 states this prototype already covers,
computes a representative centroid (lat/lon) for each district, rounds
coordinate precision to keep the file small and fast to load, and writes:

  public/india-districts.geojson   (map polygons — filtered + simplified)
  data/districts.json              (flat list: state, district, lat, lon, codes)
"""
import json
from shapely.geometry import shape

SRC = "/tmp/test.json"
OUT_GEOJSON = "/home/claude/forecastguard/public/india-districts.geojson"
OUT_DISTRICTS = "/home/claude/forecastguard/data/districts.json"

# The 18 states already used across the prototype (dashboard, alerts, etc.)
PROJECT_STATES = {
    "Rajasthan", "Gujarat", "Maharashtra", "Madhya Pradesh", "Uttar Pradesh",
    "Odisha", "West Bengal", "Assam", "Kerala", "Tamil Nadu", "Karnataka",
    "Delhi", "Bihar", "Punjab", "Andhra Pradesh", "Telangana", "Jharkhand",
    "Chhattisgarh",
}


def round_coords(geom_dict, ndigits=3):
    """Round polygon coordinate precision to shrink file size (~3 decimal
    places is ~100m precision, plenty for a state/district-level demo map)."""
    def _round(obj):
        if isinstance(obj, list):
            if obj and isinstance(obj[0], (int, float)):
                return [round(v, ndigits) for v in obj]
            return [_round(o) for o in obj]
        return obj
    geom_dict["coordinates"] = _round(geom_dict["coordinates"])
    return geom_dict


def main():
    with open(SRC) as f:
        data = json.load(f)

    kept_features = []
    districts = []
    seen = set()
    used_codes = set()

    for feat in data["features"]:
        props = feat["properties"]
        state = props.get("st_nm")
        if state not in PROJECT_STATES:
            continue

        district = props.get("district")
        if not district:
            continue
        dt_code = props.get("dt_code")
        key = (state, district)
        if key in seen:
            continue
        seen.add(key)

        try:
            geom = shape(feat["geometry"])
            centroid = geom.centroid
            lon, lat = round(centroid.x, 4), round(centroid.y, 4)
        except Exception:
            continue

        state_code = "IN-" + "".join(w[0] for w in state.split())[:2].upper()
        # Build a unique code per state: try 3-letter prefix first, then widen
        # or add a numeric suffix on collision so no two districts ever collide.
        candidate = f"{state_code}-{district[:3].upper()}"
        n = 4
        while candidate in used_codes and n <= len(district):
            candidate = f"{state_code}-{district[:n].upper()}"
            n += 1
        suffix = 2
        while candidate in used_codes:
            candidate = f"{state_code}-{district[:3].upper()}{suffix}"
            suffix += 1
        district_code = candidate
        used_codes.add(district_code)

        districts.append({
            "state": state,
            "state_code": state_code,
            "district": district,
            "district_code": district_code,
            "dt_code_source": dt_code,
            "latitude": lat,
            "longitude": lon,
        })

        feat["properties"] = {
            "state": state,
            "district": district,
            "district_code": district_code,
        }
        feat["geometry"] = round_coords(feat["geometry"], ndigits=3)
        kept_features.append(feat)

    out_geo = {"type": "FeatureCollection", "features": kept_features}
    with open(OUT_GEOJSON, "w") as f:
        json.dump(out_geo, f, separators=(",", ":"))  # compact, no pretty-print

    with open(OUT_DISTRICTS, "w") as f:
        json.dump(districts, f, indent=2)

    print(f"Kept {len(kept_features)} districts across {len(PROJECT_STATES)} states")
    per_state = {}
    for d in districts:
        per_state[d["state"]] = per_state.get(d["state"], 0) + 1
    for s, c in sorted(per_state.items()):
        print(f"  {s}: {c} districts")


if __name__ == "__main__":
    main()
