#!/usr/bin/env python3
import json
import csv

INPUT_JSON = "filtered_training_data.json"
OUTPUT_CSV = "goal_cmd.csv"

def main():
    # Load JSON
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    rows = []

    for item in data:
        goal = item.get("goal", [])
        cmd  = item.get("command", [])

        # goal = [x, y]
        gx = goal[0] if len(goal) > 0 else None
        gy = goal[1] if len(goal) > 1 else None

        # command = [v, w]
        v  = cmd[0] if len(cmd) > 0 else None
        w  = cmd[1] if len(cmd) > 1 else None

        rows.append([gx, gy, v, w])

    # Save CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["goal_x", "goal_y", "cmd_v", "cmd_w"])
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

