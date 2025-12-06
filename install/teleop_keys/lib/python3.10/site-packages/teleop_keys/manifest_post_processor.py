import os
import json

INPUT_FOLDER = "manifest_files"   # <-- change this
OUTPUT_JSON = "filtered_training_data.json"


def process_json_file(filepath):
    filename = os.path.basename(filepath)
    prefix = os.path.splitext(filename)[0]  # e.g., straight_run.json → straight_run

    with open(filepath, "r") as f:
        data = json.load(f)

    filtered_entries = []

    for entry in data:
        command = entry.get("command", [])

        # Keep only entries where at least one command element is non-zero
        if any(abs(c) > 1e-6 for c in command):
            # Modify image_path
            old_path = entry.get("image_path", "")

            if old_path.startswith("rgb_frames/"):
                parts = old_path.split("/", 1)
                entry["image_path"] = f"rgb_frames/{prefix}/{parts[1]}"
            else:
                # If path doesn't follow expected format, still prepend the prefix
                entry["image_path"] = f"rgb_frames/{prefix}/{old_path}"

            filtered_entries.append(entry)

    return filtered_entries


def main():
    all_filtered = []

    for file in os.listdir(INPUT_FOLDER):
        if file.endswith(".json"):
            path = os.path.join(INPUT_FOLDER, file)
            print(f"Processing {file}...")
            entries = process_json_file(path)
            all_filtered.extend(entries)

    print(f"Total filtered entries: {len(all_filtered)}")

    # Save combined output
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_filtered, f, indent=2)

    print(f"Saved filtered dataset to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()

