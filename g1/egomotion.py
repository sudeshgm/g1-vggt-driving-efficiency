"""Ego-motion state per window. Thresholds from idea data/NOTES.md:
stationary: mean speed < 0.5 m/s; straight: speed > 5 m/s and |yaw rate| < 2 deg/s; turning: |yaw rate| > 5 deg/s.
Anything else is 'other' (slow-moving or mild curve) and is reported, not forced into a class."""
import numpy as np
from .geometry import quat_to_R, yaw_of_R

def ego_stats(ego_records):
    t = np.array([e["timestamp"] for e in ego_records]) * 1e-6
    xyz = np.array([e["translation"] for e in ego_records])
    yaw = np.unwrap([yaw_of_R(quat_to_R(e["rotation"])) for e in ego_records])
    dur = t[-1] - t[0]
    seg = np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1)
    speed = seg.sum() / dur
    # forward speed: displacement projected on heading
    head = np.stack([np.cos(yaw[:-1]), np.sin(yaw[:-1])], 1)
    fwd = (np.diff(xyz[:, :2], axis=0) * head).sum(1).sum() / dur
    yaw_rate = np.degrees(yaw[-1] - yaw[0]) / dur
    max_yaw_rate = np.degrees(np.max(np.abs(np.diff(yaw) / np.diff(t))))
    return dict(duration_s=dur, mean_speed=speed, forward_speed=fwd,
                yaw_rate_dps=yaw_rate, max_abs_yaw_rate_dps=max_yaw_rate)

def label(stats):
    s, yr = stats["mean_speed"], abs(stats["yaw_rate_dps"])
    if s < 0.5: return "stationary"
    if yr > 5: return "turning"
    if s > 5 and yr < 2: return "straight"
    return "other"
