"""Minimal nuScenes table reader (no devkit dependency)."""
import json, os
from collections import defaultdict
import numpy as np
from .geometry import T_from_record, yaw_of_R

class NuMini:
    def __init__(self, root, version="v1.0-mini"):
        self.root = root
        def L(n): return json.load(open(os.path.join(root, version, n + ".json")))
        self.scene = L("scene"); self.sample = {r["token"]: r for r in L("sample")}
        self.sd = {r["token"]: r for r in L("sample_data")}
        self.cs = {r["token"]: r for r in L("calibrated_sensor")}
        self.ego = {r["token"]: r for r in L("ego_pose")}
        self.sensor = {r["token"]: r for r in L("sensor")}
        self.channel_of_cs = {k: self.sensor[v["sensor_token"]]["channel"] for k, v in self.cs.items()}
        self.by_scene_channel = defaultdict(list)
        for r in self.sd.values():
            ch = self.channel_of_cs[r["calibrated_sensor_token"]]
            scene = self.sample[r["sample_token"]]["scene_token"]
            self.by_scene_channel[(scene, ch)].append(r)
        for v in self.by_scene_channel.values(): v.sort(key=lambda r: r["timestamp"])

    def stream(self, scene_token, channel):
        return self.by_scene_channel[(scene_token, channel)]

    def path(self, sd): return os.path.join(self.root, sd["filename"])
    def K(self, sd): return np.array(self.cs[sd["calibrated_sensor_token"]]["camera_intrinsic"])
    def cam_cs(self, sd): return self.cs[sd["calibrated_sensor_token"]]
    def ego_pose(self, sd): return self.ego[sd["ego_pose_token"]]

    def cam_to_world(self, sd):
        return T_from_record(self.ego_pose(sd)) @ T_from_record(self.cam_cs(sd))

    def nearest(self, stream, ts):
        tss = np.array([r["timestamp"] for r in stream])
        i = int(np.argmin(np.abs(tss - ts)))
        return stream[i], abs(int(tss[i]) - ts)
