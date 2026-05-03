#!/usr/bin/env python3
# Align /rgb (Image), /scan_lidar (LaserScan), cmd_vel, and local_goal by time.
import os, csv, math, json, numpy as np
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

# --- image saving deps ---
from cv_bridge import CvBridge
import cv2
# -------------------------

BAG = "synced_bag"  # <-- folder containing .db3/.mcap
RGB_TOPIC         = "/rgb"
LIDAR_TOPIC       = "/scan_lidar"
CMD_TOPIC         = "/spyder/cmd_vel_synced"      # Int32MultiArray OR Twist OR TwistStamped
CMD_STAMP_TOPIC   = "/spyder/cmd_vel_stamp"       # builtin_interfaces/Time (optional)

GOAL_TOPIC        = "/local_goal"                 # geometry_msgs/Pose2D
GOAL_STAMP_TOPIC  = "/local_goal_stamp"           # builtin_interfaces/Time

OUT_CSV       = "aligned_samples_with_tf.csv"     # name kept, but TF removed
OUT_LIDAR_CSV = "lidar_ranges.csv"

MAX_DT_RGB_CMD_NS   = 50_000_000   # 50 ms tolerance
MAX_DT_RGB_LIDAR_NS = 50_000_000
MAX_DT_RGB_GOAL_NS  = 50_000_000

RGB_SAVE_DIR = "rgb_frames"
os.makedirs(RGB_SAVE_DIR, exist_ok=True)
bridge = CvBridge()

def to_ns(sec, nsec): 
    return int(sec)*1_000_000_000 + int(nsec)

def nearest(query_ns, ts):
    if not ts:
        return (-1, math.inf)
    lo, hi = 0, len(ts)-1
    if query_ns <= ts[0]:
        return (0, ts[0]-query_ns)
    if query_ns >= ts[-1]:
        return (hi, query_ns-ts[-1])
    while lo <= hi:
        mid = (lo+hi)//2
        if ts[mid] == query_ns:
            return (mid, 0)
        if ts[mid] < query_ns:
            lo = mid+1
        else:
            hi = mid-1
    left  = (hi, abs(ts[hi]-query_ns))
    right = (lo, abs(ts[lo]-query_ns))
    return left if left[1] <= right[1] else right

def detect_storage_id(path):
    if os.path.isdir(path):
        for f in os.listdir(path):
            if f.endswith(".mcap"):
                return "mcap"
            if f.endswith(".db3"):
                return "sqlite3"
    return "mcap" if path.endswith(".mcap") else "sqlite3"

def main():
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=BAG, storage_id=detect_storage_id(BAG)),
        ConverterOptions("", "")
    )

    topics = reader.get_all_topics_and_types()
    type_map  = {c.name: get_message(c.type) for c in topics}
    type_name = {c.name: c.type for c in topics}

    rgb_ts, lidar_ts = [], []
    lidar_pairs = []

    # command structures
    cmd_bagt, cmd_arrays = [], []
    stamp_bagt, stamp_ns_vals = [], []   # for /spyder/cmd_vel_stamp or TwistStamped

    # goal structures
    goal_bagt, goal_arrays = [], []
    goal_stamp_bagt, goal_stamp_ns_vals = [], []  # for /local_goal_stamp

    have_cmd        = CMD_TOPIC in type_map
    have_stamp      = CMD_STAMP_TOPIC in type_map
    have_goal       = GOAL_TOPIC in type_map
    have_goal_stamp = GOAL_STAMP_TOPIC in type_map

    while reader.has_next():
        topic, data, bag_t_ns = reader.read_next()

        if topic == RGB_TOPIC:
            msg = deserialize_message(data, type_map[topic])
            t_ns = to_ns(msg.header.stamp.sec, msg.header.stamp.nanosec)
            rgb_ts.append(t_ns)
            try:
                cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                cv2.imwrite(os.path.join(RGB_SAVE_DIR, f"{t_ns}.jpg"), cv_img)
            except Exception as e:
                print(f"[WARN] Failed to save frame {t_ns}: {e}")

        elif topic == LIDAR_TOPIC:
            msg = deserialize_message(data, type_map[topic])  # sensor_msgs/msg/LaserScan
            t_ns = to_ns(msg.header.stamp.sec, msg.header.stamp.nanosec)
            lidar_ts.append(t_ns)
            try:
                lidar_pairs.append((t_ns, [float(r) for r in msg.ranges]))
            except Exception:
                lidar_pairs.append((t_ns, []))

        # ------------------ CMD handling ------------------
        elif have_cmd and topic == CMD_TOPIC:
            tn  = type_name[topic]
            msg = deserialize_message(data, type_map[topic])

            if tn.endswith("/Int32MultiArray"):
                cmd_vec = list(msg.data)
                ts_ns_from_header = None
            elif tn.endswith("/Twist"):
                cmd_vec = [msg.linear.x, msg.angular.z]
                ts_ns_from_header = None
            elif tn.endswith("/TwistStamped"):
                cmd_vec = [
                    msg.twist.linear.x,  msg.twist.linear.y,  msg.twist.linear.z,
                    msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z
                ]
                ts_ns_from_header = to_ns(msg.header.stamp.sec, msg.header.stamp.nanosec)
            else:
                cmd_vec = []
                ts_ns_from_header = None

            cmd_bagt.append(bag_t_ns)
            cmd_arrays.append(cmd_vec)

            if ts_ns_from_header is not None:
                stamp_bagt.append(bag_t_ns)
                stamp_ns_vals.append(ts_ns_from_header)

        elif have_stamp and topic == CMD_STAMP_TOPIC:
            msg = deserialize_message(data, type_map[topic])  # builtin_interfaces/Time
            stamp_bagt.append(bag_t_ns)
            stamp_ns_vals.append(to_ns(msg.sec, msg.nanosec))

        # ------------------ GOAL handling -----------------
        elif have_goal and topic == GOAL_TOPIC:
            # geometry_msgs/Pose2D
            msg = deserialize_message(data, type_map[topic])
            goal_vec = [float(msg.x), float(msg.y), float(msg.theta)]
            goal_bagt.append(bag_t_ns)
            goal_arrays.append(goal_vec)

        elif have_goal_stamp and topic == GOAL_STAMP_TOPIC:
            # builtin_interfaces/Time
            msg = deserialize_message(data, type_map[topic])
            goal_stamp_bagt.append(bag_t_ns)
            goal_stamp_ns_vals.append(to_ns(msg.sec, msg.nanosec))

    # --------- Build cmd_time_ns / cmd_vals ----------
    stamp_bagt_sorted_idx = np.argsort(stamp_bagt)
    stamp_bagt     = [stamp_bagt[i] for i in stamp_bagt_sorted_idx]
    stamp_ns_vals  = [stamp_ns_vals[i] for i in stamp_bagt_sorted_idx]

    cmd_bagt_sorted_idx = np.argsort(cmd_bagt)
    cmd_bagt   = [cmd_bagt[i] for i in cmd_bagt_sorted_idx]
    cmd_arrays = [cmd_arrays[i] for i in cmd_bagt_sorted_idx]

    cmd_time_ns, cmd_vals = [], []
    if stamp_bagt:
        # use explicit stamps (TwistStamped or /cmd_vel_stamp)
        for bt, arr in zip(cmd_bagt, cmd_arrays):
            idx, _ = nearest(bt, stamp_bagt)
            if idx != -1:
                cmd_time_ns.append(stamp_ns_vals[idx])
                cmd_vals.append(arr)
    else:
        # fall back to bag time
        for bt, arr in zip(cmd_bagt, cmd_arrays):
            cmd_time_ns.append(bt)
            cmd_vals.append(arr)

    # --------- Build goal_time_ns / goal_vals ----------
    goal_stamp_bagt_sorted_idx = np.argsort(goal_stamp_bagt)
    goal_stamp_bagt    = [goal_stamp_bagt[i] for i in goal_stamp_bagt_sorted_idx]
    goal_stamp_ns_vals = [goal_stamp_ns_vals[i] for i in goal_stamp_bagt_sorted_idx]

    goal_bagt_sorted_idx = np.argsort(goal_bagt)
    goal_bagt   = [goal_bagt[i] for i in goal_bagt_sorted_idx]
    goal_arrays = [goal_arrays[i] for i in goal_bagt_sorted_idx]

    goal_time_ns, goal_vals = [], []
    if goal_stamp_bagt:
        # pair /local_goal bag times with /local_goal_stamp times
        for bt, arr in zip(goal_bagt, goal_arrays):
            idx, _ = nearest(bt, goal_stamp_bagt)
            if idx != -1:
                goal_time_ns.append(goal_stamp_ns_vals[idx])
                goal_vals.append(arr)
    else:
        # no stamp topic → use bag time
        for bt, arr in zip(goal_bagt, goal_arrays):
            goal_time_ns.append(bt)
            goal_vals.append(arr)

    # --------- Sort other streams ----------
    rgb_ts.sort()
    lidar_ts.sort()
    lidar_pairs.sort(key=lambda p: p[0])

    # --------- Align everything to RGB timestamps ----------
    rows = [(
        "rgb_ts_ns",
        "lidar_ts_ns",
        "cmd_ts_ns",
        "cmd_vals",
        "goal_ts_ns",
        "goal"          # list [x, y, theta]
    )]

    for t_img in rgb_ts:
        # LiDAR
        li, dtl = nearest(t_img, lidar_ts)
        lidar_t = lidar_ts[li] if li != -1 and dtl <= MAX_DT_RGB_LIDAR_NS else ""

        # Command
        ci, dtc = nearest(t_img, cmd_time_ns)
        cmd_t = cmd_time_ns[ci] if ci != -1 and dtc <= MAX_DT_RGB_CMD_NS else ""
        cmd_v = cmd_vals[ci] if cmd_t != "" else []

        # Goal
        gi, dtg = nearest(t_img, goal_time_ns)
        goal_t = goal_time_ns[gi] if gi != -1 and dtg <= MAX_DT_RGB_GOAL_NS else ""
        goal_v = goal_vals[gi] if goal_t != "" else []

        rows.append((t_img, lidar_t, cmd_t, cmd_v, goal_t, goal_v))

    # --------- Write CSVs ----------
    with open(OUT_CSV, "w", newline="") as f:
        csv.writer(f).writerows(rows)

    with open(OUT_LIDAR_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("lidar_ts_ns", "lidar_ranges"))
        for t_ns, ranges in lidar_pairs:
            w.writerow((t_ns, json.dumps(ranges)))

    print(f"Wrote {len(rows)-1} aligned rows (RGB+LiDAR+cmd+goal) to {OUT_CSV}")
    print(f"Wrote {len(lidar_pairs)} LiDAR rows to {OUT_LIDAR_CSV}")
    print(f"Saved {len(rgb_ts)} RGB frames under ./{RGB_SAVE_DIR}")


if __name__ == "__main__":
    main()

