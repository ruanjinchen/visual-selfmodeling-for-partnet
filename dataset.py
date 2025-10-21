# -*- coding: utf-8 -*-
import os
import re
import math
import json
import torch
import numpy as np
from torch.utils.data import Dataset

_IDX_RE = re.compile(r'^mesh_(\d+)\.xyzn$', re.IGNORECASE)

def _parse_idx_from_path(p: str) -> str:
    """mesh_123.xyzn -> '123'；Windows/Linux 路径均安全。"""
    base = os.path.basename(p)
    m = _IDX_RE.match(base)
    if not m:
        raise ValueError(f"[dataset] 无法从文件名解析编号：{base}")
    return m.group(1)

class MultipleModel(Dataset):
    """
    单关节（DOF=1）数据集：
      - 输入：'coords' (N, 3)，'states' (N, 1)  —— 角度已除以 π
      - 监督：'sdf' (N, 1)（on-surface=0，off-surface=-1），'normals' (N, 3)

    pointcloud_folder 目录应包含：
      - mesh_*.xyzn（每行 x y z nx ny nz；若只有 xyz 也可）
      - robot_state.json（键为 '编号'，值为 [[angle, 0.0]]）
    训练/验证划分来自：./assets/datainfo/multiple_models_data_split_dict_{seed}.json

    额外参数：
      - preload: bool，是否在 __init__ 一次性读入所有 mesh 并缓存（默认 True）
    """
    def __init__(self, flag, seed, pointcloud_folder, on_surface_points, preload=True, **_):
        super().__init__()
        self.flag = str(flag)
        self.seed = int(seed)
        self.pointcloud_folder = str(pointcloud_folder)
        self.on_surface_points = int(on_surface_points)
        self.preload = bool(preload)

        # 读取 robot_state.json（单关节；仅 angle）
        self.robot_state_dict = self.load_robot_state()
        # 构建划分文件列表（仅保留两边都存在的样本）
        self.all_filelist = self.get_all_filelist()

        if len(self.all_filelist) == 0:
            raise RuntimeError(
                f"[dataset] 划分 '{self.flag}' 为空，检查 {self.pointcloud_folder} 及 assets/datainfo/multiple_models_data_split_dict_{self.seed}.json"
            )

        # 预计算每个样本的 angle/π
        self._angle_map = {}
        miss_state = 0
        for mesh_path in self.all_filelist:
            k = _parse_idx_from_path(mesh_path)
            if k not in self.robot_state_dict:
                miss_state += 1
                self._angle_map[k] = np.float32(0.0)
            else:
                self._angle_map[k] = np.float32(self.robot_state_dict[k][0][0] / math.pi)
        if miss_state > 0:
            print(f"[dataset][warn] 有 {miss_state} 个样本在 robot_state.json 中缺失键；其 angle 置为 0。")

        # 预加载 mesh（可选）
        self._coords_cache = None
        self._normals_cache = None
        if self.preload:
            coords_cache = []
            normals_cache = []
            for i, path in enumerate(self.all_filelist):
                c, n = self._load_and_normalize(path)
                coords_cache.append(c.astype(np.float32, copy=False))
                normals_cache.append(n.astype(np.float32, copy=False))
            self._coords_cache = coords_cache
            self._normals_cache = normals_cache
            print(f"[dataset] 预加载完成：{len(self.all_filelist)} 个 mesh 已缓存到内存。")

    # ---------- 文件与 split ----------
    def load_robot_state(self):
        robot_state_filepath = os.path.join(self.pointcloud_folder, 'robot_state.json')
        if not os.path.exists(robot_state_filepath):
            raise FileNotFoundError(f"[dataset] 未找到 robot_state.json: {robot_state_filepath}")
        with open(robot_state_filepath, 'r', encoding='utf-8') as f:
            robot_state_dict = json.load(f)
        return robot_state_dict

    def get_all_filelist(self):
        split_json = os.path.join('.', 'assets', 'datainfo', f'multiple_models_data_split_dict_{self.seed}.json')
        if not os.path.exists(split_json):
            raise FileNotFoundError(f"[dataset] 未找到数据划分文件：{split_json}")
        with open(split_json, 'r', encoding='utf-8') as file:
            seq_dict = json.load(file)
        if self.flag not in seq_dict:
            raise KeyError(f"[dataset] 划分键 '{self.flag}' 不在 multiple_models_data_split_dict_{self.seed}.json 中。")

        id_lst = seq_dict[self.flag]
        filelist = []
        miss_mesh = 0
        for idx in id_lst:
            mesh_path = os.path.join(self.pointcloud_folder, f'mesh_{idx}.xyzn')
            if not os.path.exists(mesh_path):
                miss_mesh += 1
                continue
            # robot_state 必须也有
            if str(idx) not in self.robot_state_dict:
                continue
            filelist.append(mesh_path)

        if miss_mesh > 0:
            print(f"[dataset][warn] 划分 '{self.flag}' 中有 {miss_mesh} 个 mesh_*.xyzn 缺失，已跳过。")
        return filelist

    # ---------- 读点 + 归一 ----------
    def _load_and_normalize(self, filepath: str):
        # 用 loadtxt（比 genfromtxt 快），同时兼容只有 xyz 的情况
        pc = np.loadtxt(filepath, dtype=np.float32)
        if pc.ndim == 1:
            pc = pc.reshape(1, -1)
        if pc.shape[1] >= 6:
            coords = pc[:, :3].astype(np.float32, copy=False)
            normals = pc[:, 3:6].astype(np.float32, copy=False)
        elif pc.shape[1] == 3:
            coords = pc[:, :3].astype(np.float32, copy=False)
            normals = np.zeros_like(coords, dtype=np.float32)
        else:
            raise ValueError(f"[dataset] {filepath} 列数异常：{pc.shape}")

        # 与原项目保持一致的几何归一（提高采样效率；会形变但训练一致）
        coords = coords.copy()
        coords[:, 0] = coords[:, 0] / 0.45
        coords[:, 1] = coords[:, 1] / 0.45
        coords[:, 2] = coords[:, 2] - 0.13
        coords[:, 2] = (coords[:, 2] + 0.13) / (0.51 + 0.13)
        coords[:, 2] = coords[:, 2] - 0.5
        coords[:, 2] = coords[:, 2] / 0.5
        return coords, normals

    def __len__(self):
        return len(self.all_filelist)

    # ---------- 单样本 ----------
    def __getitem__(self, idx):
        if self._coords_cache is not None:
            coords = self._coords_cache[idx]
            normals = self._normals_cache[idx]
        else:
            coords, normals = self._load_and_normalize(self.all_filelist[idx])

        on_n = self.on_surface_points
        off_n = on_n
        total_n = on_n + off_n

        N = coords.shape[0]
        # 如果点数不够，允许重复采样，避免报错
        rand_idcs = np.random.choice(N, size=on_n, replace=(N < on_n))
        on_coords = coords[rand_idcs, :]
        on_normals = normals[rand_idcs, :]

        off_coords = np.random.uniform(-1, 1, size=(off_n, 3)).astype(np.float32)
        off_normals = np.ones((off_n, 3), dtype=np.float32) * -1.0

        final_coords = np.concatenate((on_coords, off_coords), axis=0).astype(np.float32, copy=False)
        final_normals = np.concatenate((on_normals, off_normals), axis=0).astype(np.float32, copy=False)

        sdf = np.zeros((total_n, 1), dtype=np.float32)
        sdf[on_n:, :] = -1.0

        idx_str = _parse_idx_from_path(self.all_filelist[idx])
        angle = self._angle_map.get(idx_str, np.float32(0.0))
        states = np.full((total_n, 1), angle, dtype=np.float32)

        return (
            {'coords': torch.from_numpy(final_coords), 'states': torch.from_numpy(states)},
            {'sdf': torch.from_numpy(sdf), 'normals': torch.from_numpy(final_normals)}
        )

class MultipleModelLink(Dataset):
    """保留原接口的占位；如未使用 kinematic 训练流程，可忽略。"""
    def __init__(self, flag, seed, pointcloud_folder, **kwargs):
        super().__init__()
        self.flag = str(flag)
        self.seed = int(seed)
        self.pointcloud_folder = str(pointcloud_folder)
        self.all_filelist = self.get_all_filelist()
        self.robot_state_dict = self.load_robot_state()

    def get_all_filelist(self):
        filelist = []
        if self.flag == 'val':
            for idx in range(10000, 11000):
                filepath = os.path.join(self.pointcloud_folder, f'mesh_{idx}.xyzn')
                if os.path.exists(filepath):
                    filelist.append(filepath)
        else:
            split_json = os.path.join('.', 'assets', 'datainfo', f'multiple_models_data_split_dict_{self.seed}.json')
            if not os.path.exists(split_json):
                raise FileNotFoundError(f"[dataset] 未找到数据划分文件：{split_json}")
            with open(split_json, 'r', encoding='utf-8') as file:
                seq_dict = json.load(file)
            id_lst = seq_dict[self.flag]
            for idx in id_lst:
                filepath = os.path.join(self.pointcloud_folder, f'mesh_{idx}.xyzn')
                if os.path.exists(filepath):
                    filelist.append(filepath)
        return filelist

    def __len__(self):
        return len(self.all_filelist)

    def __getitem__(self, idx):
        idx_str = _parse_idx_from_path(self.all_filelist[idx])
        robot_state = self.robot_state_dict[idx_str]  # [[angle, 0.0]]
        angle = float(robot_state[0][0]) / math.pi
        sel_robot_state = np.array([angle], dtype=np.float32)
        return {'states': torch.from_numpy(sel_robot_state).float()}, {'target_states': torch.from_numpy(sel_robot_state).float()}

    def load_robot_state(self):
        if self.flag == 'val':
            robot_state_filepath = os.path.join(self.pointcloud_folder, 'robot_state_kinematic_val.json')
            if not os.path.exists(robot_state_filepath):
                robot_state_filepath = os.path.join(self.pointcloud_folder, 'robot_state.json')
        else:
            robot_state_filepath = os.path.join(self.pointcloud_folder, 'robot_state.json')
        with open(robot_state_filepath, 'r', encoding='utf-8') as file:
            robot_state_dict = json.load(file)
        return robot_state_dict
