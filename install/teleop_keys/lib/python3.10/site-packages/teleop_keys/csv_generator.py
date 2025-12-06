#!/usr/bin/env python3
# Align /rgb (Image), /scan_lidar (LaserScan), /tf (TFMessage), and cmd_vel (Int32MultiArray+stamp) by time.
import os, csv, math, json, numpy as np
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

# --- image saving deps ---
from cv_bridge import CvBridge
import cv2
# -------------------------

BAG = "synced_bag"  # <-- folder containing .db3
RGB_TOPIC = "/rgb"
LIDAR_TOPIC = "/scan_lidar"
TF_TOPIC = "/tf"
CMD_TOPIC = "/spyder/cmd_vel_synced"   # Int32MultiArray
CMD_STAMP_TOPIC = "/spyder/cmd_vel_stamp"  # builtin_interfaces/Time

OUT_CSV = "aligned_samples_with_tf.csv"       # (unchanged)
OUT_LIDAR_CSV = "lidar_ranges.csv"            # NEW: separate LiDAR CSV

MAX_DT_RGB_CMD_NS   = 50_000_000   # 50 ms tolerance
MAX_DT_RGB_LIDAR_NS = 50_000_000   # 50 ms tolerance
MAX_DT_RGB_TF_NS    = 50_000_000   # 50 ms tolerance

# where to save image frames
RGB_SAVE_DIR = "rgb_frames"
os.makedirs(RGB_SAVE_DIR, exist_ok=True)
bridge = CvBridge()

def to_ns(sec, nsec): return int(sec)*1_000_000_000 + int(nsec)

def nearest(query_ns, ts):
    if not ts: return (-1, math.inf)
    lo, hi = 0, len(ts)-1
    if query_ns <= ts[0]: return (0, ts[0]-query_ns)
    if query_ns >= ts[-1]: return (hi, query_ns-ts[-1])
    while lo <= hi:
        mid = (lo+hi)//2
        if ts[mid] == query_ns: return (mid, 0)
        if ts[mid] < query_ns: lo = mid+1
        else: hi = mid-1
    left  = (hi, abs(ts[hi]-query_ns))
    right = (lo, abs(ts[lo]-query_ns))
    return left if left[1] <= right[1] else right

def detect_storage_id(path):
    if os.path.isdir(path):
        for f in os.listdir(path):
            if f.endswith(".mcap"): return "mcap"
            if f.endswith(".db3"):  return "sqlite3"
    return "mcap" if path.endswith(".mcap") else "sqlite3"

def main():
    reader = SequentialReader()
    reader.open(StorageOptions(uri=BAG, storage_id=detect_storage_id(BAG)),
                ConverterOptions("", ""))

    type_map = { c.name: get_message(c.type) for c in reader.get_all_topics_and_types() }

    rgb_ts, lidar_ts = [], []
    # NEW: store lidar (timestamp, ranges) pairs for the separate CSV
    lidar_pairs = []  # list of tuples: (t_ns, [ranges...])

    cmd_bagt, cmd_arrays, stamp_bagt, stamp_ns_vals = [], [], [], []
    tf_ts, tf_data = [], []

    while reader.has_next():
        topic, data, bag_t_ns = reader.read_next()

        if topic == RGB_TOPIC:
            msg = deserialize_message(data, type_map[topic])
            t_ns = to_ns(msg.header.stamp.sec, msg.header.stamp.nanosec)
            rgb_ts.append(t_ns)

            # save image with filename = timestamp in ns
            try:
                cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                img_path = os.path.join(RGB_SAVE_DIR, f"{t_ns}.jpg")
                cv2.imwrite(img_path, cv_img)
            except Exception as e:
                print(f"[WARN] Failed to save frame {t_ns}: {e}")

        elif topic == LIDAR_TOPIC:
            msg = deserialize_message(data, type_map[topic])  # sensor_msgs/msg/LaserScan
            t_ns = to_ns(msg.header.stamp.sec, msg.header.stamp.nanosec)
            lidar_ts.append(t_ns)
            # collect ranges for separate CSV
            try:
                lidar_pairs.append((t_ns, [float(r) for r in msg.ranges]))
            except Exception:
                lidar_pairs.append((t_ns, []))

        elif topic == TF_TOPIC:
            msg = deserialize_message(data, type_map[topic])  # tf2_msgs/TFMessage
            for tr in msg.transforms:
                t_ns = to_ns(tr.header.stamp.sec, tr.header.stamp.nanosec)
                tf_ts.append(t_ns)
                tf_data.append({
                    "parent": tr.header.frame_id,
                    "child": tr.child_frame_id,
                    "translation": [tr.transform.translation.x, tr.transform.translation.y, tr.transform.translation.z],
                    "rotation": [tr.transform.rotation.x, tr.transform.rotation.y, tr.transform.rotation.z, tr.transform.rotation.w]
                })

        elif topic == CMD_TOPIC:
            msg = deserialize_message(data, type_map[topic])  # std_msgs/Int32MultiArray
            cmd_bagt.append(bag_t_ns)
            cmd_arrays.append(list(msg.data))

        elif topic == CMD_STAMP_TOPIC:
            msg = deserialize_message(data, type_map[topic])  # builtin_interfaces/Time
            stamp_bagt.append(bag_t_ns)
            stamp_ns_vals.append(to_ns(msg.sec, msg.nanosec))

    # Sort and pair command + timestamps
    stamp_bagt_sorted_idx = np.argsort(stamp_bagt)
    stamp_bagt = [stamp_bagt[i] for i in stamp_bagt_sorted_idx]
    stamp_ns_vals = [stamp_ns_vals[i] for i in stamp_bagt_sorted_idx]

    cmd_bagt_sorted_idx = np.argsort(cmd_bagt)
    cmd_bagt = [cmd_bagt[i] for i in cmd_bagt_sorted_idx]
    cmd_arrays = [cmd_arrays[i] for i in cmd_bagt_sorted_idx]

    cmd_time_ns, cmd_vals = [], []
    for bt, arr in zip(cmd_bagt, cmd_arrays):
        idx, _ = nearest(bt, stamp_bagt)
        if idx != -1:
            cmd_time_ns.append(stamp_ns_vals[idx])
            cmd_vals.append(arr)

    # Sort other times
    rgb_ts.sort()
    lidar_ts.sort()
    # sort lidar_pairs by timestamp as well
    lidar_pairs.sort(key=lambda p: p[0])

    tf_ts_sorted_idx = np.argsort(tf_ts)
    tf_ts = [tf_ts[i] for i in tf_ts_sorted_idx]
    tf_data = [tf_data[i] for i in tf_ts_sorted_idx]

    # Align everything to RGB timestamps (unchanged)
    rows = [("rgb_ts_ns","lidar_ts_ns","cmd_ts_ns","cmd_vals","tf_ts_ns","tf_parent","tf_child","tf_translation","tf_rotation")]
    for t_img in rgb_ts:
        # LiDAR
        li, dtl = nearest(t_img, lidar_ts)
        lidar_t = lidar_ts[li] if li != -1 and dtl <= MAX_DT_RGB_LIDAR_NS else ""

        # Command
        ci, dtc = nearest(t_img, cmd_time_ns)
        cmd_t = cmd_time_ns[ci] if ci != -1 and dtc <= MAX_DT_RGB_CMD_NS else ""
        cmd_v = cmd_vals[ci] if cmd_t != "" else []

        # TF
        ti, dtt = nearest(t_img, tf_ts)
        if ti != -1 and dtt <= MAX_DT_RGB_TF_NS:
            tf_entry = tf_data[ti]
            tf_t = tf_ts[ti]
            tf_parent, tf_child = tf_entry["parent"], tf_entry["child"]
            tf_trans, tf_rot = tf_entry["translation"], tf_entry["rotation"]
        else:
            tf_t = tf_parent = tf_child = ""
            tf_trans = tf_rot = []

        rows.append((t_img, lidar_t, cmd_t, cmd_v, tf_t, tf_parent, tf_child, tf_trans, tf_rot))

    # Write main aligned CSV (unchanged)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerows(rows)

    # Write separate LiDAR CSV (timestamp + ranges as JSON)
    with open(OUT_LIDAR_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("lidar_ts_ns", "lidar_ranges"))
        for t_ns, ranges in lidar_pairs:
            w.writerow((t_ns, json.dumps(ranges)))

    print(f"Wrote {len(rows)-1} aligned rows with TF data to {OUT_CSV}")
    print(f"Wrote {len(lidar_pairs)} LiDAR rows to {OUT_LIDAR_CSV}")
    print(f"Saved {len(rgb_ts)} RGB frames under ./{RGB_SAVE_DIR}")

if __name__ == "__main__":
    main()

