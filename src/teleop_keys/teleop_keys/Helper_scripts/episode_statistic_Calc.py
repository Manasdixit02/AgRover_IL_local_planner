import ast
import math
import numpy as np
import pandas as pd


def load_cmd_vel(df: pd.DataFrame, col: str = "cmd_vals"):
    """
    Parses cmd_vals from string like "[v, w]" into two float arrays v and w.
    """
    cmd = df[col].apply(ast.literal_eval)
    v = cmd.apply(lambda x: float(x[0]) if len(x) > 0 else float("nan")).to_numpy()
    w = cmd.apply(lambda x: float(x[1]) if len(x) > 1 else 0.0).to_numpy()
    return v, w


def compute_duration_seconds(df: pd.DataFrame, ts_col: str = "rgb_ts_ns") -> float:
    """
    Duration = (last_timestamp - first_timestamp) in seconds.
    Assumes timestamps are in nanoseconds.
    """
    t = df[ts_col].to_numpy(dtype=np.float64) * 1e-9
    return float(t[-1] - t[0])


def compute_distance_m(df: pd.DataFrame, ts_col: str = "cmd_ts_ns", cmd_col: str = "cmd_vals") -> float:
    """
    Distance traveled ≈ Σ v_i * Δt_i (zero-order hold on v).
    """
    v, _w = load_cmd_vel(df, cmd_col)
    t = df[ts_col].to_numpy(dtype=np.float64) * 1e-9
    dt = np.diff(t)

    v_i = v[:-1]  # velocity applied over [i, i+1]
    return float(np.sum(v_i * dt))


def compute_displacement_m(
    df: pd.DataFrame,
    ts_col: str = "cmd_ts_ns",
    cmd_col: str = "cmd_vals",
    x0: float = 0.0,
    y0: float = 0.0,
    theta0: float = 0.0,
):
    """
    Dead-reckoning displacement from (v, w) using unicycle model:
      theta_{k+1} = theta_k + w_k dt
      x_{k+1} = x_k + v_k cos(theta_k) dt
      y_{k+1} = y_k + v_k sin(theta_k) dt

    Returns:
      displacement_m, (xf, yf, thetaf)
    """
    v, w = load_cmd_vel(df, cmd_col)
    t = df[ts_col].to_numpy(dtype=np.float64) * 1e-9
    dt = np.diff(t)

    x, y, theta = x0, y0, theta0

    # Apply command i over the interval [i, i+1]
    for vi, wi, dti in zip(v[:-1], w[:-1], dt):
        x += float(vi) * math.cos(theta) * float(dti)
        y += float(vi) * math.sin(theta) * float(dti)
        theta += float(wi) * float(dti)

    displacement = math.hypot(x - x0, y - y0)
    return float(displacement), (float(x), float(y), float(theta))


if __name__ == "__main__":
    csv_path = "aligned_samples_with_tf.csv"  # change if needed
    df = pd.read_csv(csv_path)

    duration_s = compute_duration_seconds(df, ts_col="rgb_ts_ns")
    distance_m = compute_distance_m(df, ts_col="cmd_ts_ns", cmd_col="cmd_vals")
    displacement_m, (xf, yf, thetaf) = compute_displacement_m(df, ts_col="cmd_ts_ns", cmd_col="cmd_vals")

    print(f"Total duration:      {duration_s:.6f} s")
    print(f"Distance traveled:   {distance_m:.6f} m")
    print(f"Final pose (dead-reckoned): x={xf:.6f}, y={yf:.6f}, theta={thetaf:.6f} rad")
    print(f"Displacement:        {displacement_m:.6f} m")

