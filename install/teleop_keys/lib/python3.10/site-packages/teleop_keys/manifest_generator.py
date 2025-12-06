#!/usr/bin/env python3
import csv, json, os, sys
from typing import List, Optional

# --------- CONFIG ----------
CSV_PATH = "aligned_expanded_relpose_rt.csv"
LIDAR_CSV_PATH = "lidar_ranges.csv"   # read lidar ranges from this file
OUTPUT_JSON = "manifest.json"
IMAGE_DIR = "rgb_frames"              # image path becomes rgb_frames/<timestamp>.jpg
IMAGE_EXT = ".jpg"
# ---------------------------

def _is_empty(x) -> bool:
    return x is None or str(x).strip() == ""

def _parse_list_field(val: str) -> List[float]:
    """
    Parse a list-like field that may look like:
      "[1.0, 2.0, 3.0]"  or  "1.0,2.0,3.0"  or  "1.0 2.0 3.0"
    Returns a list of floats. Empty/invalid -> [].
    """
    if val is None:
        return []
    s = str(val).strip()
    if s == "":
        return []
    # Try JSON first
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [float(x) for x in parsed]
    except Exception:
        pass
    # Fallback: split by commas or whitespace
    sep = "," if "," in s else None
    try:
        return [float(x) for x in s.split(sep) if x.strip() != ""]
    except Exception:
        return []

def _normalize_ts(ts_value) -> Optional[str]:
    """
    Canonicalize a timestamp to an integer-ns string, so "17304...", "1.7304e19",
    "17304.0" all become "17304...".
    Returns None if it cannot parse.
    """
    if _is_empty(ts_value):
        return None
    s = str(ts_value).strip()
    try:
        # float first to handle scientific notation; then int to drop decimals
        v = int(float(s))
        return str(v)
    except Exception:
        # last-ditch: keep only digits
        digits = "".join(ch for ch in s if ch.isdigit())
        return digits if digits else None

def _collect_goal(row: dict) -> Optional[List[float]]:
    """
    Try multiple goal encodings:
      1) 'goal' as a list-like string
      2) polar: ('goal_r','goal_theta')   <-- original
         ALSO accept ('r','theta')        <-- NEW
      3) cartesian+heading: ('goal_dx','goal_dy','goal_dtheta')
      4) cartesian only: ('goal_dx','goal_dy')
    """
    # 1) single field list
    for key in ("goal", "goal_vec", "goal_relpose"):
        if key in row and not _is_empty(row[key]):
            lst = _parse_list_field(row[key])
            if lst:
                return lst

    # 2) polar with goal_ prefix
    if all(k in row for k in ("goal_r", "goal_theta")) and not _is_empty(row["goal_r"]) and not _is_empty(row["goal_theta"]):
        try:
            return [float(row["goal_r"]), float(row["goal_theta"])]
        except Exception:
            pass

    # 2b) polar with bare names r, theta  <-- NEW
    if all(k in row for k in ("r", "theta")) and not _is_empty(row["r"]) and not _is_empty(row["theta"]):
        try:
            return [float(row["r"]), float(row["theta"])]
        except Exception:
            pass

    # 3) cartesian + heading
    keys3 = ("goal_dx", "goal_dy", "goal_dtheta")
    if all(k in row and not _is_empty(row[k]) for k in keys3):
        try:
            return [float(row["goal_dx"]), float(row["goal_dy"]), float(row["goal_dtheta"])]
        except Exception:
            pass

    # 4) cartesian only
    keys2 = ("goal_dx", "goal_dy")
    if all(k in row and not _is_empty(row[k]) for k in keys2):
        try:
            return [float(row["goal_dx"]), float(row["goal_dy"])]
        except Exception:
            pass

    return None

# NOTE: kept here to "change nothing else" structurally, but we won't use this anymore.
def _collect_lidar(_row: dict) -> Optional[List[float]]:
    return None

def _collect_command(row: dict) -> Optional[List[float]]:
    """
    Accept:
      - single list field: 'command', 'cmd', 'cmd_vals'
      - scalar columns: ('cmd_vx','cmd_vy','cmd_wz') or ('vx','vy','wz') or ('v_x','v_y','w_z')
    """
    for key in ("command", "cmd", "cmd_vals"):
        if key in row and not _is_empty(row[key]):
            lst = _parse_list_field(row[key])
            if lst:
                return lst

    triplets = [
        ("cmd_vx", "cmd_vy", "cmd_wz"),
        ("vx", "vy", "wz"),
        ("v_x", "v_y", "w_z"),
    ]
    for trip in triplets:
        if all(k in row and not _is_empty(row[k]) for k in trip):
            try:
                return [float(row[trip[0]]), float(row[trip[1]]), float(row[trip[2]])]
            except Exception:
                continue
    return None

def _load_lidar_map(path: str) -> dict:
    """
    Load lidar_ranges.csv into a dict: { normalized_lidar_ts_ns(str) : List[float] }
    Expects columns: lidar_ts_ns, lidar_ranges (ranges stored as JSON).
    """
    if not os.path.exists(path):
        print(f"ERROR: {path} not found.")
        sys.exit(1)
    m = {}
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        if "lidar_ts_ns" not in r.fieldnames or "lidar_ranges" not in r.fieldnames:
            print("ERROR: lidar_ranges.csv must have columns: lidar_ts_ns, lidar_ranges")
            sys.exit(1)
        for row in r:
            key_raw = row.get("lidar_ts_ns", "")
            key = _normalize_ts(key_raw)
            if not key:
                continue
            ranges = _parse_list_field(row.get("lidar_ranges", ""))
            m[key] = ranges
    return m

def main():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found.")
        sys.exit(1)

    # preload lidar map from the separate lidar CSV
    lidar_map = _load_lidar_map(LIDAR_CSV_PATH)

    entries = []
    with open(CSV_PATH, "r", newline="") as f:
        reader = csv.DictReader(f)
        missing_cols = [c for c in ("rgb_ts_ns", "lidar_ts_ns") if c not in reader.fieldnames]
        if missing_cols:
            print(f"ERROR: CSV must have columns: rgb_ts_ns, lidar_ts_ns. Missing: {missing_cols}")
            sys.exit(1)

        for row in reader:
            rgb_ts_raw = row.get("rgb_ts_ns", "")
            lidar_ts_raw = row.get("lidar_ts_ns", "")

            rgb_ts = _normalize_ts(rgb_ts_raw)
            lidar_ts = _normalize_ts(lidar_ts_raw)

            # Keep only rows having both stamps
            if not rgb_ts or not lidar_ts:
                continue

            # Build image path: rgb_frames/<timestamp>.jpg
            image_path = os.path.join(IMAGE_DIR, f"{rgb_ts}{IMAGE_EXT}")

            # LiDAR: read by normalized lidar_ts_ns from lidar_ranges.csv
            lidar = lidar_map.get(lidar_ts, None)
            if lidar is None or len(lidar) == 0:
                continue

            # Goal
            goal = _collect_goal(row)
            if goal is None:
                continue

            # Command
            command = _collect_command(row)
            if command is None:
                continue

            entries.append({
                "image_path": image_path.replace("\\", "/"),
                "lidar": lidar,
                "goal": goal,
                "command": command
            })

    # Write JSON list
    with open(OUTPUT_JSON, "w") as jf:
        json.dump(entries, jf, indent=2)

    print(f"Wrote {len(entries)} samples to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()

