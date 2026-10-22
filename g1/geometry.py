"""Rigid transforms, LiDAR->camera projection. Conventions follow nuScenes:
quaternions are [w,x,y,z]; calibrated_sensor gives sensor->ego; ego_pose gives ego->global."""
import numpy as np

def quat_to_R(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

def make_T(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t; return T

def T_from_record(rec):
    return make_T(quat_to_R(rec["rotation"]), rec["translation"])

def lidar_to_cam_T(lidar_cs, lidar_ego, cam_cs, cam_ego):
    """4x4 mapping lidar-frame points (at lidar timestamp) into camera frame (at camera timestamp),
    compensating ego motion through the global frame."""
    T_l2e = T_from_record(lidar_cs); T_e2g_l = T_from_record(lidar_ego)
    T_c2e = T_from_record(cam_cs); T_e2g_c = T_from_record(cam_ego)
    return np.linalg.inv(T_c2e) @ np.linalg.inv(T_e2g_c) @ T_e2g_l @ T_l2e

def project_points(pts_cam, K, hw, min_depth=1.0):
    """pts_cam (N,3) in camera frame (z forward). Returns u,v (int pixel), depth, for points in image."""
    H, W = hw
    z = pts_cam[:, 2]
    m = z > min_depth
    p = pts_cam[m]
    uv = (K @ p.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    u = np.floor(uv[:, 0]).astype(int); v = np.floor(uv[:, 1]).astype(int)
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u[inb], v[inb], p[inb, 2]

def sparse_depth_map(pts_cam, K, hw, min_depth=1.0):
    """Z-buffered sparse depth (0 = no LiDAR). Nearest point wins per pixel."""
    H, W = hw
    u, v, d = project_points(pts_cam, K, hw, min_depth)
    D = np.full((H, W), np.inf, np.float32)
    order = np.argsort(-d)              # far first, near overwrites
    D[v[order], u[order]] = d[order]
    D[np.isinf(D)] = 0
    return D

def load_lidar_bin(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :3]

def transform(T, pts):
    return pts @ T[:3, :3].T + T[:3, 3]

def yaw_of_R(R):
    return np.arctan2(R[1, 0], R[0, 0])
