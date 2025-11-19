
import os
import glob
import math
import json
import torch
import numpy as np
from torch.utils.data import Dataset

class MultipleModel(Dataset):
    def __init__(self, flag, seed, pointcloud_folder, on_surface_points, dof=4, cache_to='none'):
        super().__init__()

        self.flag = flag
        self.seed = seed
        self.pointcloud_folder = pointcloud_folder
        self.on_surface_points = on_surface_points
        self.dof = dof
        self.all_filelist = self.get_all_filelist()
        self.robot_state_dict = self.load_robot_state()
        self.cache_to = cache_to
        self.use_cache = False
        if self.cache_to == 'cpu':
            self._preload_all()

    def _fast_read_xyzn(self, path):
        npy = path + '.npy'
        if os.path.exists(npy):
            return np.load(npy)
        arr = np.loadtxt(path)          # 比 genfromtxt 快很多
        np.save(npy, arr)               # 首次运行落盘缓存
        return arr

    def _preload_all(self):
        self.cache_coords, self.cache_normals = [], []
        for path in self.all_filelist:
            arr = self._fast_read_xyzn(path)
            c = torch.from_numpy(arr[:, :3]).float()
            n = torch.from_numpy(arr[:, 3:]).float()
            # 放入共享内存，供 DataLoader 的多个 worker 共享
            c.share_memory_(); n.share_memory_()
            self.cache_coords.append(c); self.cache_normals.append(n)
        self.use_cache = True

    def get_all_filelist(self):
        filelist = []
        with open(os.path.join('assets', 'datainfo', f'multiple_models_data_split_dict_{self.seed}.json'), 'r') as file:
            seq_dict = json.load(file)
        id_lst = seq_dict[self.flag]

        for idx in id_lst:
            filepath = os.path.join(self.pointcloud_folder, f'mesh_{idx}.xyzn')
            filelist.append(filepath)
        return filelist

    def __len__(self):
        return len(self.all_filelist)

    def __getitem__(self, idx):

        # =====> sdf
        # coords, normals = self.load_pcd(idx)
        if self.use_cache:
            coords = self.cache_coords[idx]     # torch.FloatTensor (CPU)
            normals = self.cache_normals[idx]
        else:
            coords_np, normals_np = self.load_pcd(idx)
            coords = torch.from_numpy(coords_np).float()
            normals = torch.from_numpy(normals_np).float()
        off_surface_samples = self.on_surface_points  # **2
        total_samples = self.on_surface_points + off_surface_samples

        # random coords
        # point_cloud_size = coords.shape[0]
        # rand_idcs = np.random.choice(point_cloud_size, size=self.on_surface_points)

        # on_surface_coords = coords[rand_idcs, :]
        # on_surface_normals = normals[rand_idcs, :]

        # off_surface_coords = np.random.uniform(-1, 1, size=(off_surface_samples, 3))
        # off_surface_normals = np.ones((off_surface_samples, 3)) * -1
        # 用 torch 在 CPU 上采样，避免 numpy/torch 频繁来回
        point_cloud_size = coords.shape[0]
        rand_idcs = torch.randint(point_cloud_size, (self.on_surface_points,))
        on_surface_coords = coords.index_select(0, rand_idcs)
        on_surface_normals = normals.index_select(0, rand_idcs)
        off_surface_coords = torch.rand(off_surface_samples, 3) * 2 - 1
        off_surface_normals = torch.full((off_surface_samples, 3), -1.0)

        # sdf = np.zeros((total_samples, 1))  # on-surface = 0
        # sdf[self.on_surface_points:, :] = -1  # off-surface = -1

        # final_coords = np.concatenate((on_surface_coords, off_surface_coords), axis=0)
        # final_normals = np.concatenate((on_surface_normals, off_surface_normals), axis=0)
        sdf = torch.zeros((total_samples, 1)); sdf[self.on_surface_points:, :] = -1
        final_coords = torch.cat((on_surface_coords, off_surface_coords), dim=0)
        final_normals = torch.cat((on_surface_normals, off_surface_normals), dim=0)

        # =====> robot state
        index = self.all_filelist[idx].split('/')[-1].split('.')[0].split('_')[1]
        robot_state = self.robot_state_dict[index]
        # sel_robot_state = np.array([robot_state[k][0] for k in range(self.dof)], dtype=np.float32)
        # sel_robot_state = sel_robot_state / np.pi
        # sel_robot_state = sel_robot_state.reshape(1, -1)
        # final_robot_states = np.tile(sel_robot_state, (total_samples, 1))
        sel = np.array([robot_state[k][0] for k in range(self.dof)], dtype=np.float32) / np.pi
        sel = torch.from_numpy(sel).view(1, -1).repeat(total_samples, 1).float()

        return {'coords': final_coords.float(), 'states': sel.float()}, \
               {'sdf': sdf.float(), 'normals': final_normals.float()}
    
    def load_pcd(self, idx):
        xyzn_txt = self.all_filelist[idx]

        xyzn_npy = xyzn_txt + ".npy"
        if os.path.exists(xyzn_npy):
            point_cloud = np.load(xyzn_npy)  # 直接二进制载入
        else:
            point_cloud = np.loadtxt(xyzn_txt)   # 比 genfromtxt 快
            np.save(xyzn_npy, point_cloud)       # 首次运行后落盘缓存
            # print("首次运行 落盘缓存")

        coords = point_cloud[:, :3]
        normals = point_cloud[:, 3:]

        # reshape point cloud such that it lies in bounding box of (-1, 1) (distorts geometry, but makes for high sample efficiency)
        # coords[:, 0] = coords[:, 0] / 0.45 # (-1, 1)
        # coords[:, 1] = coords[:, 1] / 0.45 # (-1, 1)
        # coords[:, 2] = coords[:, 2] - 0.13 # zero centering (-0.13, 0.51)
        # coords[:, 2] = (coords[: ,2] + 0.13) / (0.51 + 0.13) # (0, 1)
        # coords[:, 2] = coords[:, 2] - 0.5 # (-0.5, 0.5)
        # coords[:, 2] = coords[:, 2] / 0.5 # (-1, 1)
        return coords, normals
    
    def load_robot_state(self):
        robot_state_filepath = os.path.join(self.pointcloud_folder, 'robot_state.json')
        with open(robot_state_filepath, 'r') as file:
            robot_state_dict = json.load(file)
        return robot_state_dict

class MultipleModelLink(Dataset):
    def __init__(self, flag, seed, pointcloud_folder):
        super().__init__()

        self.flag = flag
        self.seed = seed
        self.pointcloud_folder = pointcloud_folder
        self.all_filelist = self.get_all_filelist()
        self.robot_state_dict = self.load_robot_state()

    def get_all_filelist(self):
        filelist = []
        if self.flag == 'val':
            for idx in range(10000, 11000):
                filepath = os.path.join(self.pointcloud_folder, f'mesh_{idx}.xyzn')
                filelist.append(filepath)
        else:
            with open(os.path.join('../assets', 'datainfo', f'multiple_models_data_split_dict_{self.seed}.json'), 'r') as file:
                seq_dict = json.load(file)
            id_lst = seq_dict[self.flag]

            for idx in id_lst:
                filepath = os.path.join(self.pointcloud_folder, f'mesh_{idx}.xyzn')
                filelist.append(filepath)
        return filelist

    def __len__(self):
        return len(self.all_filelist)

    def __getitem__(self, idx):

        # =====> robot state
        index = self.all_filelist[idx].split('/')[-1].split('.')[0].split('_')[1]
        robot_state = self.robot_state_dict[index]
        sel_robot_state = np.array([robot_state[0][0], robot_state[1][0], robot_state[2][0], robot_state[3][0]])
        tar_robot_state = np.array([robot_state[5][0], robot_state[5][1], robot_state[5][2]])
        sel_robot_state = sel_robot_state / np.pi

        return {'states': torch.from_numpy(sel_robot_state).float()},{'target_states': torch.from_numpy(tar_robot_state).float()}
    
    def load_robot_state(self):
        if self.flag == 'val':
            robot_state_filepath = os.path.join(self.pointcloud_folder, 'robot_state_kinematic_val.json')
        else:
            robot_state_filepath = os.path.join(self.pointcloud_folder, 'robot_state.json')
        with open(robot_state_filepath, 'r') as file:
            robot_state_dict = json.load(file)
        return robot_state_dict