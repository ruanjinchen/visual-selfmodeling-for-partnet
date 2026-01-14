# -*- coding: utf-8 -*-
"""
批量导出 *测试集* 的所有关节 condition 的预测（mesh 与点云，PLY）及对应 GT（mesh 与点云，PLY）。
- 测试集来源：从 split JSON（multiple_models_data_split_dict_*.json）中的 "test" 读取
- 坐标系：严格不做任何缩放/反归一化；工作与导出均在训练使用的全局单位球坐标（默认 AABB=[-1,1]^3）
- 文件命名：
    gt/gt_000000.ply,      gt/gt_mesh_000000.ply
    pred/pred_000000.ply,  pred/pred_mesh_000000.ply
- 不计算任何 CD/EMD

用法示例：
python export_test_all_states_ply.py \
  --ckpt /path/to/checkpoint.ckpt \
  --data-dir /path/to/dataset_dir \
  --split-json /path/to/multiple_models_data_split_dict_1.json \
  --out-dir export_out \
  --res 160 --iso 0.0 --device cuda
"""
import os
import json
import math
import argparse
from pathlib import Path
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

# ===== 你的模型（与 demo 一致）=====
from models import VisModelingModel  # 项目内模块

# ---------- 工具函数 ----------
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
    # 常见子模块名
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

def parse_state_vec(entry) -> np.ndarray:
    """
    将 robot_state.json 的 value 解析为关节角向量（单位：弧度）。
    支持：
      - [[angle, ...], [angle2, ...], ...]  -> 取每项第一个数
      - [angle1, angle2, ...]               -> 直接转换
      - {j0:[angle,...], j1:[angle2,...]}   -> 按键名排序取每项第一个数
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

def discover_test_keys_from_split(data_dir: Path, rs_keys_sorted, split_json: str | None):
    """
    读取 split JSON 的 "test" 列表（保序），若不可用则回退到其他探测方式。
    优先检查：
      1) --split-json 指定的文件
      2) data_dir 下匹配 multiple_models_data_split_dict*.json
    仍不可用时回退到：
      - data_dir/splits/test_ids.txt
      - data_dir/test_ids.txt
      - data_dir/splits.json or split.json 的 'test'
      - 否则使用 rs_keys_sorted 全量
    """
    # 1) 指定路径
    cand_paths = []
    if split_json:
        cand_paths.append(Path(split_json))
    # 2) data_dir 下常见命名（包含你提供的 multiple_models_data_split_dict_1.json）
    for name in [
        "multiple_models_data_split_dict.json",
        "multiple_models_data_split_dict_1.json",
        "multiple_models_data_split_dict_0.json",
    ]:
        p = data_dir / name
        if p.exists():
            cand_paths.append(p)

    # 去重保持顺序
    seen = set()
    cand_paths = [p for p in cand_paths if not (str(p) in seen or seen.add(str(p)))]

    # 尝试读取 split json
    for p in cand_paths:
        try:
            with p.open('r', encoding='utf-8') as f:
                obj = json.load(f)
            if isinstance(obj, dict) and 'test' in obj and isinstance(obj['test'], list):
                test_ids = [str(x) for x in obj['test']]
                # 仅保留 robot_state.json 中实际存在的 key，并保持 split 顺序
                rs_set = set(str(k) for k in rs_keys_sorted)
                filtered = [k for k in test_ids if k in rs_set]
                if len(filtered) == 0:
                    print(f"[warn] {p} 的 'test' 与 robot_state.json 无交集，回退到其它方式。")
                else:
                    print(f"[info] 使用 split: {p} 中的 'test'（共 {len(filtered)} 个）。")
                    return filtered
        except Exception as e:
            print(f"[warn] 读取 split 文件失败：{p} ({e})，继续尝试其它来源。")

    # 其它本地分割文件
    cands_txt = [data_dir / 'splits' / 'test_ids.txt', data_dir / 'test_ids.txt']
    for p in cands_txt:
        if p.exists():
            with p.open('r', encoding='utf-8') as f:
                ids = [line.strip() for line in f if line.strip()]
            return [k for k in rs_keys_sorted if k in ids]

    cands_json = [data_dir / 'splits.json', data_dir / 'split.json']
    for p in cands_json:
        if p.exists():
            with p.open('r', encoding='utf-8') as f:
                o = json.load(f)
            if isinstance(o, dict) and 'test' in o:
                ids = [str(x) for x in o['test']]
                return [k for k in rs_keys_sorted if k in ids]

    # 回退：全量
    return rs_keys_sorted

def find_gt_mesh_path(data_dir: Path, key: str):
    """在数据目录中尝试找到 GT 网格文件（若不存在则返回 None）。"""
    bases = [f"mesh_{key}", f"model_{key}", f"{key}"]
    exts = ['.ply', '.obj', '.off', '.stl', '.glb', '.gltf']
    for b in bases:
        for e in exts:
            p = data_dir / f"{b}{e}"
            if p.exists():
                return p
    return None

def load_gt_points(data_dir: Path, key: str) -> np.ndarray:
    """
    读取 GT 点云：优先 data_dir/mesh_{key}.xyzn.npy（取前三列为 xyz），
    其次 data_dir/mesh_{key}.xyzn（文本，取前三列为 xyz）；
    若不存在则尝试从 GT 网格采样。
    """
    xyzn = data_dir / f"mesh_{key}.xyzn"
    xyzn_npy = data_dir / f"mesh_{key}.xyzn.npy"
    xyz = data_dir / f"mesh_{key}.xyz"
    xyz_npy = data_dir / f"mesh_{key}.xyz.npy"

    # 1) 优先读取二进制 npy（更快，也与训练时的数据格式一致）
    if xyzn_npy.exists():
        arr = np.load(str(xyzn_npy)).astype(np.float32, copy=False)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr[:, :3]

    # 2) 兼容文本版 .xyzn
    if xyzn.exists():
        arr = np.loadtxt(str(xyzn), dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr[:, :3].astype(np.float32, copy=False)

    # 3) 兼容仅有 xyz 的情况（如果你未来想省空间）
    if xyz_npy.exists():
        arr = np.load(str(xyz_npy)).astype(np.float32, copy=False)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr[:, :3]
    if xyz.exists():
        arr = np.loadtxt(str(xyz), dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr[:, :3].astype(np.float32, copy=False)

    gt_mesh_path = find_gt_mesh_path(data_dir, key)
    if gt_mesh_path is not None and HAS_TRIMESH:
        m = trimesh.load(gt_mesh_path, force='mesh', process=False)
        if not isinstance(m, trimesh.Trimesh):
            raise RuntimeError(f"无法将 {gt_mesh_path} 作为网格加载")
        pts, _ = trimesh.sample.sample_surface(m, 100000)
        return pts.astype(np.float32)
    raise FileNotFoundError(
        f"未找到 {xyzn_npy} 或 {xyzn}（也未找到 {xyz_npy}/{xyz}），且找不到可加载的 GT 网格用于采样。"
    )

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

@torch.no_grad()
def query_volume_state(model: torch.nn.Module,
                       state_vec: np.ndarray,
                       res: int,
                       device: str,
                       aabb_min: float,
                       aabb_max: float,
                       chunk: int = 262144):
    """
    用给定关节状态向量评估整个体素网格；仅对状态向量做 angle/π 的数值缩放。
    """
    pts = grid_points(res, aabb_min=aabb_min, aabb_max=aabb_max)  # (N,3)
    st = (state_vec.astype(np.float32) / math.pi).reshape(1, -1)
    states = np.repeat(st, repeats=pts.shape[0], axis=0).astype(np.float32)
    coords_t = torch.from_numpy(pts).to(device, non_blocking=True)
    states_t = torch.from_numpy(states).to(device, non_blocking=True)

    sdf_chunks = []
    for i in range(0, pts.shape[0], chunk):
        sdf = call_model_sdf(model, coords_t[i:i+chunk], states_t[i:i+chunk])
        sdf_chunks.append(sdf.squeeze(-1).float().cpu().numpy())
    vol = np.concatenate(sdf_chunks, axis=0).reshape(res, res, res)  # (z,y,x)
    return vol

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

def maybe_clean_components(verts: np.ndarray, faces: np.ndarray,
                           keep_largest: bool, min_comp_ratio: float,
                           min_comp_verts: int, min_comp_faces: int):
    """
    可选的连通块清理（默认关闭，不影响坐标尺度）。
    """
    if not HAS_TRIMESH or not (keep_largest or min_comp_ratio > 0 or
                               min_comp_verts > 0 or min_comp_faces > 0):
        return verts, faces
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    parts = mesh.split(only_watertight=False)
    if not parts:
        return verts, faces
    areas = np.array([max(m.area, 1e-9) for m in parts], dtype=float)
    if keep_largest:
        m = parts[int(np.argmax(areas))]
        return m.vertices.view(np.ndarray), m.faces.view(np.ndarray)
    thr_abs = float(min_comp_ratio) * float(areas.sum()) if min_comp_ratio > 0 else 0.0
    def keep(m: trimesh.Trimesh, a: float):
        if thr_abs > 0.0 and a < thr_abs: return False
        if min_comp_verts > 0 and len(m.vertices) < int(min_comp_verts): return False
        if min_comp_faces > 0 and len(m.faces)   < int(min_comp_faces):  return False
        return True
    kept = [m for m, a in zip(parts, areas) if keep(m, a)]
    if not kept:
        m = parts[int(np.argmax(areas))]
        return m.vertices.view(np.ndarray), m.faces.view(np.ndarray)
    mesh2 = trimesh.util.concatenate(kept)
    return mesh2.vertices.view(np.ndarray), mesh2.faces.view(np.ndarray)

# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--data-dir', type=str, required=True)
    ap.add_argument('--out-dir', type=str, default='export_out')
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--res', type=int, default=160)
    ap.add_argument('--iso', type=str, default='0.0')  # 'auto' 或数值
    ap.add_argument('--sample-n', type=int, default=100000)
    ap.add_argument('--aabb-min', type=float, default=-1.0)  # 单位球坐标，不缩放
    ap.add_argument('--aabb-max', type=float, default= 1.0)
    ap.add_argument('--split-json', type=str, default=None,
                    help='包含 {"train":[...], "test":[...]} 的分割 JSON 路径（优先使用）。')
    # 可选去噪
    ap.add_argument('--keep-largest', action='store_true')
    ap.add_argument('--min-comp-ratio', type=float, default=0.0)
    ap.add_argument('--min-comp-verts', type=int, default=0)
    ap.add_argument('--min-comp-faces', type=int, default=0)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    pred_dir = out_dir / 'pred'
    gt_dir = out_dir / 'gt'
    pred_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    # 读取 robot_state.json
    rs_path = data_dir / 'robot_state.json'
    if not rs_path.exists():
        raise FileNotFoundError(f"未找到 {rs_path}")
    with rs_path.open('r', encoding='utf-8') as f:
        rs = json.load(f)

    # key 排序
    rs_keys_sorted = sorted(rs.keys(), key=lambda k: int(k) if str(k).isdigit() else str(k))

    # === 依据 split JSON 选择测试集（保序） ===
    test_keys = discover_test_keys_from_split(
        data_dir=data_dir,
        rs_keys_sorted=rs_keys_sorted,
        split_json=args.split_json
    )

    # 解析每个 key 的状态向量（弧度）
    state_map = {k: parse_state_vec(rs[k]) for k in test_keys}

    print(f"[info] 待导出 test 条目数：{len(test_keys)}")
    print(f"[info] AABB = [{args.aabb_min:.3f}, {args.aabb_max:.3f}]^3, res={args.res}, iso={args.iso}")

    # 加载模型
    device = args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    model = VisModelingModel.load_from_checkpoint(args.ckpt, strict=False)
    model = model.to(device).eval()

    # 导出循环
    index_map = []
    for idx, key in enumerate(test_keys):
        state_vec = state_map[key]

        # --- 体素评估（坐标不缩放） ---
        vol = query_volume_state(model,
                                 state_vec=state_vec,
                                 res=int(args.res),
                                 device=device,
                                 aabb_min=float(args.aabb_min),
                                 aabb_max=float(args.aabb_max))

        # --- ISO ---
        if args.iso.lower() == 'auto':
            # 用当前帧 GT 点在模型上的 SDF 中位数作为 iso
            gt_pts = load_gt_points(data_dir, key)
            coords_t = torch.from_numpy(gt_pts.astype(np.float32)).to(device)
            st = (state_vec.astype(np.float32) / math.pi).reshape(1, -1)
            st_rep = np.repeat(st, repeats=gt_pts.shape[0], axis=0).astype(np.float32)
            states_t = torch.from_numpy(st_rep).to(device)
            with torch.no_grad():
                iso_level = float(call_model_sdf(model, coords_t, states_t).median().item())
        else:
            try:
                iso_level = float(args.iso)
            except Exception:
                iso_level = 0.0

        # --- Marching Cubes ---
        verts, faces = marching_cubes_extract(vol,
                                              aabb_min=float(args.aabb_min),
                                              aabb_max=float(args.aabb_max),
                                              level=iso_level)
        # 可选去噪
        verts, faces = maybe_clean_components(
            verts, faces,
            keep_largest=bool(args.keep_largest),
            min_comp_ratio=float(args.min_comp_ratio),
            min_comp_verts=int(args.min_comp_verts),
            min_comp_faces=int(args.min_comp_faces)
        )

        # --- 保存预测（PLY） ---
        pred_mesh_path = pred_dir / f"pred_mesh_{idx:06d}.ply"
        pred_pts_path  = pred_dir / f"pred_{idx:06d}.ply"
        save_mesh_ply(pred_mesh_path, verts, faces)
        pred_pts = sample_surface_points(verts, faces, n=int(args.sample_n))
        save_pointcloud_ply(pred_pts_path, pred_pts)

        # --- 保存 GT（PLY） ---
        gt_pts = load_gt_points(data_dir, key)
        gt_pc_path = gt_dir / f"gt_{idx:06d}.ply"
        save_pointcloud_ply(gt_pc_path, gt_pts)

        gt_mesh_src = find_gt_mesh_path(data_dir, key)
        gt_mesh_path = gt_dir / f"gt_mesh_{idx:06d}.ply"
        if gt_mesh_src is not None and HAS_TRIMESH:
            m = trimesh.load(gt_mesh_src, force='mesh', process=False)
            if isinstance(m, trimesh.Trimesh):
                m.export(str(gt_mesh_path))
            else:
                print(f"[warn] {gt_mesh_src} 不是可用的网格，跳过 gt_mesh_{idx:06d}.ply")
        else:
            print(f"[warn] 找不到 GT 网格文件（mesh_{key}.*），已跳过 gt_mesh_{idx:06d}.ply 的导出。")

        index_map.append({
            "index": idx,
            "key": key,
            "state_rad": [float(x) for x in state_vec]
        })
        print(f"[ok] #{idx:06d} key={key} -> pred_mesh/pred_pc, gt_pc{'' if gt_mesh_src else ' (no gt mesh)'}")

    # 索引映射
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "index_map.json").open("w", encoding="utf-8") as f:
        json.dump(index_map, f, indent=2, ensure_ascii=False)

    print("\n[done] 导出完成：")
    print(f"  预测网格：   {pred_dir}/pred_mesh_XXXXXX.ply")
    print(f"  预测点云：   {pred_dir}/pred_XXXXXX.ply")
    print(f"  GT 网格：    {gt_dir}/gt_mesh_XXXXXX.ply   (若存在)")
    print(f"  GT 点云：    {gt_dir}/gt_XXXXXX.ply")
    print(f"  索引映射：   {out_dir}/index_map.json")
    print("  坐标系：     保持全局单位球坐标，不做任何缩放/反归一化。")

if __name__ == "__main__":
    main()

'''

python export_test_all_states_ply_support_npy.py \
  --ckpt /data/fllm/code/vsm/SIM2_state-condition_new-global-siren-sdf_6/lightning_logs/version_0/checkpoints/epoch=299-step=42300.ckpt \
  --data-dir dataset_tdcr/tdcr_2m_no_base_vsm \
  --split-json assets/datainfo/multiple_models_data_split_dict_6.json \
  --out-dir eval_out/sim2_no_base \
  --res 160 \
  --iso 0.0 \
  --device cuda \
  --keep-largest \
  --min-comp-verts 3000

'''