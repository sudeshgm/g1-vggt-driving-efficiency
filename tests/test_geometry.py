import numpy as np
from g1.geometry import quat_to_R, make_T, sparse_depth_map, project_points, lidar_to_cam_T, transform

def rec(q, t): return {"rotation": q, "translation": t}

def test_quat_identity_and_90z():
    assert np.allclose(quat_to_R([1, 0, 0, 0]), np.eye(3))
    R = quat_to_R([np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)])
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0])

def test_known_point_projects_to_known_pixel():
    K = np.array([[1000, 0, 800], [0, 1000, 450], [0, 0, 1.]])
    pts = np.array([[0, 0, 10.], [1, 0.5, 10.], [0, 0, -5.]])
    u, v, d = project_points(pts, K, (900, 1600))
    assert list(zip(u, v)) == [(800, 450), (900, 500)] and np.allclose(d, [10, 10])

def test_zbuffer_nearest_wins():
    K = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1.]])
    D = sparse_depth_map(np.array([[0, 0, 20.], [0, 0, 5.]]), K, (100, 100))
    assert D[50, 50] == 5 and (D > 0).sum() == 1

def test_full_chain_with_ego_motion():
    # lidar on ego at height 2 m, identity orientation; camera: optical z = ego x (forward), x=-y_ego, y=-z_ego
    Rcam = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0.]])  # cam->ego columns
    w = np.sqrt(1 + np.trace(Rcam)) / 2
    q = [w, (Rcam[2, 1]-Rcam[1, 2])/(4*w), (Rcam[0, 2]-Rcam[2, 0])/(4*w), (Rcam[1, 0]-Rcam[0, 1])/(4*w)]
    assert np.allclose(quat_to_R(q), Rcam)
    lidar_cs = rec([1, 0, 0, 0], [0, 0, 2.]); cam_cs = rec(q, [1.5, 0, 1.5])
    ego_l = rec([1, 0, 0, 0], [100, 0, 0.]); ego_c = rec([1, 0, 0, 0], [101, 0, 0.])  # ego moved 1 m forward
    # world point at (121, 0, 1.5): 20 m ahead of camera centre at x=102.5 -> depth 18.5, on principal point
    p_lidar = np.array([[121 - 100, 0, 1.5 - 2]])
    pc = transform(lidar_to_cam_T(lidar_cs, ego_l, cam_cs, ego_c), p_lidar)
    assert np.allclose(pc, [[0, 0, 18.5]])
