#!/usr/bin/env python3
import json

MANIFEST = "filtered_training_data.json"   # change if needed

def main():
    with open(MANIFEST, "r") as f:
        data = json.load(f)

    xs, ys = [], []

    for item in data:
        goal = item.get("goal", None)

        if not isinstance(goal, list) or len(goal) < 2:
            continue

        xs.append(float(goal[0]))
        ys.append(float(goal[1]))

    print("Goal X min/max:", min(xs), max(xs))
    print("Goal Y min/max:", min(ys), max(ys))

if __name__ == "__main__":
    main()

