
# -*- coding: utf-8 -*-
"""
single_demo_infer_one_state.py

用途：
- 复用 export_test_all_states_ply.py 的渲染/导出逻辑；
- 传入一组关节 joint，输出预测 mesh(.ply) 与点云(.ply)；
- 先从 checkpoint 加载模型，同时读取 config（可选）与数据集文件夹下的 robot_state.json；
- 对 robot_state.json 按“维度”统计 min/max，并提供 0-1 归一化/反归一化；
- 因训练时网络接收的是 (angle / π) 的状态输入，因此若传入的是 0-1 归一化的关节，本脚本会先“反归一化到弧度”，随后按 (rad/π) 投喂网络。

示例：
python single_demo_infer_one_state.py \
  --ckpt /path/to/xxx.ckpt \
  --data-dir /path/to/dataset_dir \
  --out-dir demo_out \
  --res 160 --iso 0.0 --device cuda \
  --joint "[0.2,0.5,0.8,0.1,0.3,0.6,0.9]" --input-norm

若需要从 robot_state.json 中某一帧直接推理：
python single_demo_infer_one_state.py --ckpt ... --data-dir ... --joint-key 842
"""
import os
import json
import math
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch

# ===== 可选依赖 =====
HAS_SKIMAGE = False
try:
    from skimage.measure import marching_cubes
    HAS_SKIMAGE = True
except Exception:
    pass

HAS_TRIMESH = False
try:
    import trimesh
    HAS_TRIMESH = True
except Exception:
    pass

from models import VisModelingModel


# ---------- 通用工具 ----------
def grid_points(res: int, aabb_min: float = -1.0, aabb_max: float = 1.0):
    xs = np.linspace(aabb_min, aabb_max, res, dtype=np.float32)
    ys = np.linspace(aabb_min, aabb_max, res, dtype=np.float32)
    zs = np.linspace(aabb_min, aabb_max, res, dtype=np.float32)
    grid_z, grid_y, grid_x = np.meshgrid(zs, ys, xs, indexing='ij')  # volume (z,y,x)
    pts = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3).astype(np.float32)  # (x,y,z)
    return pts


@torch.no_grad()
def call_model_sdf(model: torch.nn.Module, coords: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    """兼容常见前向签名，返回 (N,1) SDF。"""
    model.eval()
    # dict forward
    try:
        out = model({'coords': coords, 'states': states})
        if isinstance(out, dict):
            out = out.get('sdf', out)
        if isinstance(out, torch.Tensor):
            return out.reshape(-1, 1)
    except Exception:
        pass
    # tuple forward
    try:
        out = model(coords, states)
        if isinstance(out, dict):
            out = out.get('sdf', out)
        if isinstance(out, (list, tuple)):
            out = out[0]
        if isinstance(out, torch.Tensor):
            return out.reshape(-1, 1)
    except Exception:
        pass
    # 常见子模块名兜底
    for name in ['net', 'model', 'decoder', 'mlp', 'network', 'siren']:
        if hasattr(model, name):
            try:
                sub = getattr(model, name)
                out = sub(torch.cat([coords, states], dim=-1))
                if isinstance(out, torch.Tensor):
                    return out.reshape(-1, 1)
            except Exception:
                continue
    raise RuntimeError("请根据你的 models.py 调整 call_model_sdf() 的前向调用。")


def marching_cubes_extract(vol: np.ndarray, aabb_min: float, aabb_max: float, level=None):
    """对体素体 (z,y,x) 做 MC，输出顶点为世界系 (x,y,z) 并平移到 aabb_min 起点。"""
    vmin, vmax = float(vol.min()), float(vol.max())
    if level is None:
        if vmin <= 0.0 <= vmax: level = 0.0
        else: level = (vmax * 0.5) if vmax < 0 else (vmin * 0.5)
    res = vol.shape[0]
    step = (aabb_max - aabb_min) / (res - 1)
    spacing_zyx = (step, step, step)
    if HAS_SKIMAGE:
        verts_zyx, faces, normals, _ = marching_cubes(vol, level=level, spacing=spacing_zyx)
    else:
        import mcubes
        verts_zyx, faces = mcubes.marching_cubes(vol, level)
        verts_zyx = verts_zyx * step
    verts_xyz = np.empty_like(verts_zyx, dtype=np.float32)
    verts_xyz[:, 0] = verts_zyx[:, 2] + aabb_min  # x
    verts_xyz[:, 1] = verts_zyx[:, 1] + aabb_min  # y
    verts_xyz[:, 2] = verts_zyx[:, 0] + aabb_min  # z
    return verts_xyz, faces


def sample_surface_points(verts: np.ndarray, faces: np.ndarray, n: int = 100000):
    if HAS_TRIMESH:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        pts, _ = trimesh.sample.sample_surface(mesh, int(n))
        return pts.astype(np.float32)
    vidx = np.random.choice(len(verts), size=int(n), replace=len(verts) < n)
    return verts[vidx].astype(np.float32)


def save_mesh_ply(path: Path, verts: np.ndarray, faces: np.ndarray):
    if not HAS_TRIMESH:
        print(f"[warn] 未安装 trimesh，跳过导出：{path}")
        return
    mesh = trimesh.Trimesh(vertices=verts.astype(np.float32),
                           faces=faces.astype(np.int64),
                           process=False)
    mesh.export(str(path))  # 后缀 .ply 自动导出 PLY


def save_pointcloud_ply(path: Path, pts: np.ndarray):
    if not HAS_TRIMESH:
        print(f"[warn] 未安装 trimesh，跳过导出：{path}")
        return
    trimesh.points.PointCloud(pts.astype(np.float32)).export(str(path))


# ---------- 读取/统计 robot_state ----------
def parse_state_vec(entry) -> np.ndarray:
    """
    将 robot_state.json 的 value 解析为关节角向量（单位：弧度）。
    支持：
      - [[angle, .], [angle2, .], .]  -> 取每项第一个数
      - [angle1, angle2, .]               -> 直接转换
      - {j0:[angle,], j1:[angle2,]}   -> 按键名排序取每项第一个数
      - 单个数值                             -> 封装为长度1
    """
    if isinstance(entry, list):
        if len(entry) > 0 and isinstance(entry[0], list):
            return np.array([float(sub[0]) for sub in entry], dtype=np.float32)
        else:
            return np.array([float(x) for x in entry], dtype=np.float32)
    elif isinstance(entry, dict):
        vec = []
        for k in sorted(entry.keys()):
            v = entry[k]
            if isinstance(v, list) and len(v) > 0:
                vec.append(float(v[0]))
            else:
                vec.append(float(v))
        return np.array(vec, dtype=np.float32)
    else:
        return np.array([float(entry)], dtype=np.float32)


def load_rs_and_stats(data_dir: Path) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """读取 robot_state.json，返回：{key: state(rad)}, per-dim min(rad), per-dim max(rad)"""
    rs_path = data_dir / 'robot_state.json'
    if not rs_path.exists():
        raise FileNotFoundError(f"未找到 {rs_path}")
    with rs_path.open('r', encoding='utf-8') as f:
        rs = json.load(f)

    # 排序后的 key -> 弧度向量
    rs_keys_sorted = sorted(rs.keys(), key=lambda k: int(k) if str(k).isdigit() else str(k))
    state_map = {k: parse_state_vec(rs[k]) for k in rs_keys_sorted}

    # 维度统计
    all_vecs = np.stack(list(state_map.values()), axis=0)  # (N, D)
    vmin = np.min(all_vecs, axis=0)
    vmax = np.max(all_vecs, axis=0)
    # 防止除零
    same = np.isclose(vmax, vmin)
    vmax[same] = vmin[same] + 1.0

    return state_map, vmin.astype(np.float32), vmax.astype(np.float32)


def normalize01(rad: np.ndarray, vmin: np.ndarray, vmax: np.ndarray) -> np.ndarray:
    return (rad - vmin) / (vmax - vmin)


def denormalize01(norm01: np.ndarray, vmin: np.ndarray, vmax: np.ndarray) -> np.ndarray:
    return vmin + norm01 * (vmax - vmin)


# ---------- 推理 ----------
@torch.no_grad()
def query_volume_state(model: torch.nn.Module,
                       state_rad: np.ndarray,
                       res: int,
                       device: str,
                       aabb_min: float,
                       aabb_max: float,
                       chunk: int = 262144):
    """
    用给定关节状态（单位：弧度）评估整个体素网格；仅对状态向量做 angle/π 的数值缩放。
    """
    pts = grid_points(res, aabb_min=aabb_min, aabb_max=aabb_max)  # (N,3)
    st = (state_rad.astype(np.float32) / math.pi).reshape(1, -1)
    states = np.repeat(st, repeats=pts.shape[0], axis=0).astype(np.float32)
    coords_t = torch.from_numpy(pts).to(device, non_blocking=True)
    states_t = torch.from_numpy(states).to(device, non_blocking=True)

    sdf_chunks = []
    for i in range(0, pts.shape[0], chunk):
        sdf = call_model_sdf(model, coords_t[i:i+chunk], states_t[i:i+chunk])
        sdf_chunks.append(sdf.squeeze(-1).float().cpu().numpy())
    vol = np.concatenate(sdf_chunks, axis=0).reshape(res, res, res)  # (z,y,x)
    return vol


def read_config(config_path: str | None) -> Dict[str, Any]:
    if not config_path:
        return {}
    p = Path(config_path)
    if not p.exists():
        print(f"[warn] 未找到 config：{config_path}")
        return {}
    try:
        if p.suffix.lower() in ['.yaml', '.yml']:
            import yaml
            with p.open('r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        else:
            with p.open('r', encoding='utf-8') as f:
                return json.load(f) or {}
    except Exception as e:
        print(f"[warn] 解析 config 失败：{e}")
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True, help='Lightning checkpoint (.ckpt)')
    ap.add_argument('--data-dir', type=str, required=True, help='包含 robot_state.json 的数据目录')
    ap.add_argument('--config', type=str, default=None, help='训练/模型配置（yaml/json，可选）')
    ap.add_argument('--out-dir', type=str, default='demo_out')
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--res', type=int, default=160)
    ap.add_argument('--iso', type=str, default='0.0')  # 'auto' 或数值；此 demo 默认 0.0
    ap.add_argument('--aabb-min', type=float, default=-1.0)  # 单位球坐标，不缩放
    ap.add_argument('--aabb-max', type=float, default= 1.0)

    # 关节输入方式（二选一）：--joint 或 --joint-key
    ap.add_argument('--joint', type=str, default=None,
                    help='JSON 字符串，如 "[0.2,0.5,...]"。配合 --input-norm 表示 0-1 归一化输入。')
    ap.add_argument('--joint-key', type=str, default=None,
                    help='直接使用 robot_state.json 中的某个 key（以弧度读取）。')
    ap.add_argument('--input-norm', action='store_true', help='若传入 --joint，则表示该向量已归一化到 0-1。')

    # 可选连通块清理
    ap.add_argument('--keep-largest', action='store_true')
    ap.add_argument('--min-comp-ratio', type=float, default=0.0)
    ap.add_argument('--min-comp-verts', type=int, default=0)
    ap.add_argument('--min-comp-faces', type=int, default=0)

    # 点云采样数量
    ap.add_argument('--sample-n', type=int, default=100000)

    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 读取 config（可选）
    cfg = read_config(args.config)
    if 'aabb_min' in cfg: args.aabb_min = float(cfg['aabb_min'])
    if 'aabb_max' in cfg: args.aabb_max = float(cfg['aabb_max'])
    if 'res' in cfg:      args.res = int(cfg['res'])

    # 2) 读取 robot_state 并统计 min/max
    state_map, vmin, vmax = load_rs_and_stats(data_dir)
    stats_path = out_dir / 'state_stats.json'
    with stats_path.open('w', encoding='utf-8') as f:
        json.dump({'vmin': vmin.tolist(), 'vmax': vmax.tolist()}, f, indent=2, ensure_ascii=False)
    print(f"[info] 已统计关节维度 min/max 并保存到 {stats_path}")

    # 3) 准备待推理的关节（弧度）
    if args.joint_key is not None:
        key = str(args.joint_key)
        if key not in state_map:
            raise KeyError(f"{key} 不在 robot_state.json 中")
        state_rad = state_map[key].astype(np.float32)
        print(f"[info] 使用 robot_state.json[{key}] 的弧度向量。")
    else:
        if args.joint is None:
            raise ValueError("必须提供 --joint 或 --joint-key。")
        try:
            vec = json.loads(args.joint)
            state_in = np.array(vec, dtype=np.float32).reshape(-1)
        except Exception as e:
            raise ValueError(f"--joint 解析失败：{e}")
        if args.input_norm:
            # 反归一化到弧度
            if state_in.shape[-1] != vmin.shape[-1]:
                raise ValueError(f"--joint 维度 {state_in.shape[-1]} 与数据集关节维度 {vmin.shape[-1]} 不一致。")
            state_rad = denormalize01(state_in, vmin, vmax).astype(np.float32)
            print("[info] 已将 0-1 归一化关节向量反归一化为弧度。")
        else:
            state_rad = state_in.astype(np.float32)
            print("[info] 按弧度直接使用 --joint。")

    # 4) 加载模型
    device = args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    model = VisModelingModel.load_from_checkpoint(args.ckpt, strict=False)
    model = model.to(device).eval()

    # 5) 评估体素体（保持坐标在全局单位球）
    vol = query_volume_state(model,
                             state_rad=state_rad,
                             res=int(args.res),
                             device=device,
                             aabb_min=float(args.aabb_min),
                             aabb_max=float(args.aabb_max))

    # 6) ISO 水平
    try:
        iso_level = float(args.iso) if args.iso.lower() != 'auto' else 0.0
    except Exception:
        iso_level = 0.0

    # 7) Marching Cubes -> mesh -> 点云采样
    verts, faces = marching_cubes_extract(vol,
                                          aabb_min=float(args.aabb_min),
                                          aabb_max=float(args.aabb_max),
                                          level=iso_level)

    # 可选去噪
    if args.keep_largest or args.min_comp_ratio > 0 or args.min_comp_verts > 0 or args.min_comp_faces > 0:
        if HAS_TRIMESH:
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            parts = mesh.split(only_watertight=False)
            if parts:
                areas = np.array([max(m.area, 1e-9) for m in parts], dtype=float)
                if args.keep_largest:
                    m = parts[int(np.argmax(areas))]
                    verts, faces = m.vertices.view(np.ndarray), m.faces.view(np.ndarray)
                else:
                    thr_abs = float(args.min_comp_ratio) * float(areas.sum()) if args.min_comp_ratio > 0 else 0.0
                    def keep(m: 'trimesh.Trimesh', a: float):
                        if thr_abs > 0.0 and a < thr_abs: return False
                        if args.min_comp_verts > 0 and len(m.vertices) < int(args.min_comp_verts): return False
                        if args.min_comp_faces > 0 and len(m.faces)   < int(args.min_comp_faces):  return False
                        return True
                    kept = [m for m, a in zip(parts, areas) if keep(m, a)]
                    if kept:
                        mesh2 = trimesh.util.concatenate(kept)
                        verts, faces = mesh2.vertices.view(np.ndarray), mesh2.faces.view(np.ndarray)

    pred_mesh_path = out_dir / f"pred_mesh.ply"
    pred_pts_path  = out_dir / f"pred_points.ply"
    save_mesh_ply(pred_mesh_path, verts, faces)
    pts = sample_surface_points(verts, faces, n=int(args.sample_n))
    save_pointcloud_ply(pred_pts_path, pts)

    # 8) 也保存本次输入（归一化/反归一化）备查
    record = {
        'state_rad': state_rad.tolist(),
        'state_norm01': normalize01(state_rad, vmin, vmax).tolist(),
        'aabb': [float(args.aabb_min), float(args.aabb_max)],
        'res': int(args.res),
        'iso': float(iso_level),
    }
    with (out_dir / "infer_record.json").open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    print("\n[done] 推理完成：")
    print(f"  预测网格：   {pred_mesh_path}")
    print(f"  预测点云：   {pred_pts_path}")
    print(f"  记录文件：   {out_dir / 'infer_record.json'}")
    print(f"  关节统计：   {stats_path}")
    print("  坐标系：     保持全局单位球坐标，不做任何缩放/反归一化。")


if __name__ == "__main__":
    main()
'''
done
python single_demo_infer_one_state.py \
  --ckpt /data/fllm/code/vsm/eyeglasses_11_12_state-condition_new-global-siren-sdf_1/lightning_logs/version_2/checkpoints/epoch=299-step=4500.ckpt \
  --data-dir /data/fllm/code/vsm/data/eyeglasses_101863 \
  --out-dir draw_demo_out_1118/eyeglasses/0.970_0.162 \
  --res 160 --iso 0.0 --device cuda \
  --joint "[0.970,0.162]" \
  --input-norm \
  --min-comp-verts 100 \
  --min-comp-ratio 0.01

done
python single_demo_infer_one_state.py \
  --ckpt /data/fllm/code/vsm/pliers_11_12_state-condition_new-global-siren-sdf_1/lightning_logs/version_4/checkpoints/epoch=299-step=4500.ckpt \
  --data-dir /data/fllm/code/vsm/data/pliers_2074 \
  --out-dir draw_demo_out_1118/pliers/0.788 \
  --res 160 --iso 0.0 --device cuda \
  --joint "[0.788]" \
  --input-norm --keep-largest \
  --min-comp-verts 5000 \
  --min-comp-ratio 0.02 


done
python single_demo_infer_one_state.py \
  --ckpt /data/fllm/code/vsm/scissors_11_12_state-condition_new-global-siren-sdf_1/lightning_logs/version_5/checkpoints/epoch=299-step=4500.ckpt \
  --data-dir /data/fllm/code/vsm/data/scissors_10893 \
  --out-dir draw_demo_out_1118/scissors/0.788 \
  --res 160 --iso 0.0 --device cuda \
  --joint "[0.788]" \
  --input-norm

0.000_0.597_0.684_0.653_0.317_0.401_0.156_0.897_0.416_0.969_0.795_0.170_0.772
0.000_0.743_0.391_0.427_0.852_0.854_0.940_0.406_0.995_0.029_0.833_0.093_0.002
0.000_0.888_0.119_0.006_0.348_0.131_0.859_0.046_0.707_0.835_0.501_0.658_0.540

python single_demo_infer_one_state.py \
  --ckpt /data/fllm/code/vsm/FINAL_boston_dynamics_spot_state-condition_new-global-siren-sdf_1/lightning_logs/version_1/checkpoints/epoch=299-step=4500.ckpt \
  --data-dir data/boston_dynamics_spot_normed \
  --out-dir draw_demo_out_1118/boston_dynamics_spot/0.000_0.888_0.119_0.006_0.348_0.131_0.859_0.046_0.707_0.835_0.501_0.658_0.540_11 \
  --res 160 --iso 0.0 --device cuda \
  --joint "[0.000,0.888,0.119,0.006,0.348,0.131,0.859,0.046,0.707,0.835,0.501,0.658,0.540]" \
  --input-norm --keep-largest \
  --min-comp-verts 100 \
  --min-comp-ratio 0.01


rand0_j0.362_0.344_0.959_0.198_0.199_0.712_0.288_gt_idx00008
rand1_j0.434_0.280_0.748_0.771_0.776_0.534_0.076_gt_idx00070
rand2_j0.493_0.286_0.579_0.503_0.226_0.646_0.060_gt_idx00082

python single_demo_infer_one_state.py \
  --ckpt /data/fllm/code/vsm/franka_fr3_state-condition_new-global-siren-sdf_1/lightning_logs/version_6/checkpoints/epoch=375-step=5640.ckpt \
  --data-dir data/franka_fr3_no_scale \
  --out-dir draw_demo_out_1118/franka_fr3/0.362_0.344_0.959_0.198_0.199_0.712_0.288 \
  --res 160 --iso 0.0 --device cuda \
  --joint "[0.362,0.344,0.959,0.198,0.199,0.712,0.288]" \
  --input-norm --keep-largest \
  --min-comp-verts 100 \
  --min-comp-ratio 0.01

python single_demo_infer_one_state.py \
  --ckpt /data/fllm/code/vsm/franka_fr3_state-condition_new-global-siren-sdf_1/lightning_logs/version_6/checkpoints/epoch=375-step=5640.ckpt \
  --data-dir data/franka_fr3_no_scale \
  --out-dir draw_demo_out_1118/franka_fr3/0.434_0.280_0.748_0.771_0.776_0.534_0.076 \
  --res 160 --iso 0.0 --device cuda \
  --joint "[0.434,0.280,0.748,0.771,0.776,0.534,0.076]" \
  --input-norm --keep-largest \
  --min-comp-verts 100 \
  --min-comp-ratio 0.01

python single_demo_infer_one_state.py \
  --ckpt /data/fllm/code/vsm/franka_fr3_state-condition_new-global-siren-sdf_1/lightning_logs/version_6/checkpoints/epoch=375-step=5640.ckpt \
  --data-dir data/franka_fr3_no_scale \
  --out-dir draw_demo_out_1118/franka_fr3/0.493_0.286_0.579_0.503_0.226_0.646_0.060 \
  --res 160 --iso 0.0 --device cuda \
  --joint "[0.493,0.286,0.579,0.503,0.226,0.646,0.060]" \
  --input-norm --keep-largest \
  --min-comp-verts 100 \
  --min-comp-ratio 0.01

'''