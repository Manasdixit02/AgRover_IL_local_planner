#!/usr/bin/env python3
# Expand each row into future-relative rows and add r (distance ignoring z) and theta (yaw)
# - tf_translation: replaced with relative [dx,dy,dz]
# - tf_rotation   : replaced with relative quaternion [qx,qy,qz,qw]
# - r             : sqrt(dx^2 + dy^2)
# - theta         : yaw (rotation about Z) in radians

import csv, math, ast
import numpy as np

IN_CSV  = "aligned_samples_with_tf.csv"
OUT_CSV = "aligned_expanded_relpose_rt.csv"
STEP    = 5
HORIZON = 100

def parse_vec3(s):
    if not s or s.strip()=="":
        return None
    try:
        v = np.array(ast.literal_eval(s), dtype=float)
        if v.shape==(3,): return v
        if isinstance(v,(list,tuple)) and len(v)==3: return np.array(v,dtype=float)
    except Exception:
        pass
    return None

def parse_qxyzw(s):
    if not s or s.strip()=="":
        return None
    try:
        q = np.array(ast.literal_eval(s), dtype=float)
        if q.shape==(4,): return q
        if isinstance(q,(list,tuple)) and len(q)==4: return np.array(q,dtype=float)
    except Exception:
        pass
    return None

def quat_normalize(q):
    n = np.linalg.norm(q)
    return q if n==0 else (q / n)

def quat_conj(q):
    x,y,z,w = q
    return np.array([-x,-y,-z,w], dtype=float)

def quat_mul(q1, q2):
    x1,y1,z1,w1 = q1
    x2,y2,z2,w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ], dtype=float)

def quat_to_R(q):
    x,y,z,w = quat_normalize(q)
    xx,yy,zz = x*x, y*y, z*z
    xy,xz,yz = x*y, x*z, y*z
    wx,wy,wz = w*x, w*y, w*z
    return np.array([
        [1-2*(yy+zz),   2*(xy-wz),     2*(xz+wy)],
        [2*(xy+wz),     1-2*(xx+zz),   2*(yz-wx)],
        [2*(xz-wy),     2*(yz+wx),     1-2*(xx+yy)]
    ], dtype=float)

def yaw_from_quat(q):
    """Extract yaw (rotation about Z) from quaternion [x,y,z,w], radians."""
    x,y,z,w = quat_normalize(q)
    siny_cosp = 2.0 * (w*z + x*y)
    cosy_cosp = 1.0 - 2.0 * (y*y + z*z)
    return math.atan2(siny_cosp, cosy_cosp)

def rel_translation(p_i, q_i, p_j):
    R_i = quat_to_R(q_i)
    return R_i.T @ (p_j - p_i)

def rel_rotation(q_i, q_j):
    qi = quat_normalize(q_i)
    qj = quat_normalize(q_j)
    return quat_normalize(quat_mul(quat_conj(qi), qj))

def main():
    with open(IN_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    for col in ("r","theta"):
        if col not in fieldnames:
            fieldnames.append(col)

    poses = []
    for r in rows:
        p = parse_vec3(r.get("tf_translation",""))
        q = parse_qxyzw(r.get("tf_rotation",""))
        poses.append((p,q))

    N = len(rows)
    with open(OUT_CSV, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(N):
            p_i, q_i = poses[i]
            if p_i is None or q_i is None:
                continue

            for k in range(STEP, HORIZON+1, STEP):
                j = i + k
                if j >= N:
                    break
                p_j, q_j = poses[j]
                if p_j is None or q_j is None:
                    continue

                dt = rel_translation(p_i, q_i, p_j)
                dq = rel_rotation(q_i, q_j)
                r   = float(math.hypot(dt[0], dt[1]))     # ignore z
                th  = float(yaw_from_quat(dq))

                new_row = dict(rows[i])
                new_row["tf_translation"] = str([float(dt[0]), float(dt[1]), float(dt[2])])
                new_row["tf_rotation"]    = str([float(dq[0]), float(dq[1]), float(dq[2]), float(dq[3])])
                new_row["r"]              = r
                new_row["theta"]          = th

                writer.writerow(new_row)

    print(f"✅ Done. Wrote {OUT_CSV} with r (planar distance) and theta (yaw).")

if __name__ == "__main__":
    main()

