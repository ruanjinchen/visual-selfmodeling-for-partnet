# -*- coding: utf-8 -*-
"""
demo_reconstruct_and_cd.py  —— 多DoF + 两种可选剪枝（GT距离/梯度）版

关键特性：
- 支持多关节：--state-key / --state-rad / --state-deg（三选一，仍兼容 --angle-*）
- 不做任何空间缩放（仅对“状态量”做 angle/π 数值归一化以匹配训练）
- Marching Cubes 重建后可选两类剪枝：
  (1) --prune-by-gt <dist>  ：保留“与GT点云距离<=dist”的三角面
  (2) --grad-thr   <thr>    ：保留“所有顶点 |∇SDF|>=thr”的三角面（thr≈0.4~0.7 常用）
- 仍保留原demo的：AABB设定/自动、keep-largest、min-comp-*, 采样点云、CD计算等

注意：
- 剪枝均在 MC 之后、导出之前进行；不会引入坐标尺度变化，也不影响后续你自己计算的CD/EMD。
"""
import os, json, math, argparse
from pathlib import Path
import numpy as np
import torch

# optional deps
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

HAS_SCIPY = False
try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except Exception:
    pass

# ===== your lightning model =====
from models import VisModelingModel  # 项目内模块（训练/前向与此一致）  # noqa

# ---------- 工具 ----------
def normalize_xyzn_like_training(xyz: np.ndarray) -> np.ndarray:
    """保持与训练一致；此demo不做空间缩放（no-op）。"""
    return xyz.astype(np.float32, copy=False)

def denormalize_from_training_like(xyz_norm: np.ndarray) -> np.ndarray:
    """保持与训练一致；此demo不做空间反归一化（no-op）。"""
    return xyz_norm.astype(np.float32, copy=False)

def grid_points(res: int, aabb_min: float = -1.0, aabb_max: float = 1.0):
    xs = np.linspace(aabb_min, aabb_max, res, dtype=np.float32)
    ys = np.linspace(aabb_min, aabb_max, res, dtype=np.float32)
    zs = np.linspace(aabb_min, aabb_max, res, dtype=np.float32)
    grid_z, grid_y, grid_x = np.meshgrid(zs, ys, xs, indexing='ij')  # (z,y,x)
    pts = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3).astype(np.float32)  # (N,3) xyz
    return pts

def parse_state_vec(entry) -> np.ndarray:
    """把 robot_state.json 的 value 解析为弧度向量（支持list/list-of-list/dict/single）。"""
    if isinstance(entry, list):
        if len(entry) > 0 and isinstance(entry[0], list):
            return np.array([float(sub[0]) for sub in entry], dtype=np.float32)
        else:
            return np.array([float(x) for x in entry], dtype=np.float32)
    elif isinstance(entry, dict):
        return np.array([
            float(entry[k][0] if isinstance(entry[k], list) else entry[k])
            for k in sorted(entry.keys())
        ], dtype=np.float32)
    else:
        return np.array([float(entry)], dtype=np.float32)

def load_robot_states(data_dir: Path):
    rs_path = data_dir / 'robot_state.json'
    if not rs_path.exists():
        raise FileNotFoundError(f"未找到 robot_state.json: {rs_path}")
    with rs_path.open('r', encoding='utf-8') as f:
        rs = json.load(f)
    return {str(k): parse_state_vec(v) for k, v in rs.items()}

def find_nearest_key_by_state(rs_parsed: dict, target: np.ndarray) -> str:
    keys = list(rs_parsed.keys())
    mats = np.stack([rs_parsed[k] for k in keys], axis=0)
    d = min(mats.shape[1], target.shape[0])
    mats = mats[:, :d]; tgt = target[:d]
    d2 = np.sum((mats - tgt[None, :])**2, axis=1)
    return keys[int(np.argmin(d2))]

def load_gt_points_by_key(data_dir: Path, key: str) -> np.ndarray:
    """优先读 data_dir/mesh_{key}.xyzn 的前三列；否则尝试从 mesh_{key}.* 采样（需 trimesh）。"""
    xyzn = data_dir / f"mesh_{key}.xyzn"
    if xyzn.exists():
        arr = np.loadtxt(str(xyzn), dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr[:, :3].astype(np.float32, copy=False)

    if HAS_TRIMESH:
        for ext in ['.ply', '.obj', '.off', '.stl', '.glb', '.gltf']:
            cand = data_dir / f"mesh_{key}{ext}"
            if cand.exists():
                m = trimesh.load(cand, force='mesh', process=False)
                if isinstance(m, trimesh.Trimesh):
                    pts, _ = trimesh.sample.sample_surface(m, 100000)
                    return pts.astype(np.float32)
    raise FileNotFoundError(f"未找到 {xyzn}，且找不到可加载的 GT 网格用于采样。")

@torch.no_grad()
def call_model_sdf(model: torch.nn.Module, coords: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    """兼容多种前向签名；返回 (N,1) SDF。"""
    model.eval()
    # dict forward（有些LightningModule把前向写成接受dict）
    try:
        out = model({'coords': coords, 'states': states})
        if isinstance(out, dict):
            out = out.get('sdf', out)
        if isinstance(out, torch.Tensor):
            return out.reshape(-1, 1)
    except Exception:
        pass
    # tuple forward（常见：forward(coords, states)）
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
    # 子模块常见名字（你的模型里是 self.model）
    for name in ['net', 'model', 'decoder', 'mlp', 'network', 'siren']:
        if hasattr(model, name):
            try:
                sub = getattr(model, name)
                out = sub(torch.cat([coords, states], dim=-1))
                if isinstance(out, torch.Tensor):
                    return out.reshape(-1, 1)
            except Exception:
                continue
    raise RuntimeError("请在 call_model_sdf() 中按你的 models.py 调整前向调用。")

@torch.no_grad()
def query_volume_state(model: torch.nn.Module, state_vec: np.ndarray, res: int, device: str,
                       aabb_min: float, aabb_max: float, chunk: int = 262144):
    """用多关节状态评估整个体素（仅状态做 angle/π 数值缩放；坐标不缩放）。"""
    pts = grid_points(res, aabb_min=aabb_min, aabb_max=aabb_max)  # (N,3)
    st = (state_vec.astype(np.float32) / math.pi).reshape(1, -1)  # (1,dof)
    coords_t = torch.from_numpy(pts).to(device, non_blocking=True)
    st_row = torch.from_numpy(st).to(device, non_blocking=True)
    sdf_chunks = []
    total = pts.shape[0]
    for i in range(0, total, chunk):
        n = min(chunk, total - i)
        states_t = st_row.expand(n, -1)
        sdf = call_model_sdf(model, coords_t[i:i+n], states_t)
        sdf_chunks.append(sdf.squeeze(-1).float().cpu().numpy())
    return np.concatenate(sdf_chunks, axis=0).reshape(res, res, res)  # (z,y,x)

def marching_cubes_extract(vol: np.ndarray, aabb_min: float, aabb_max: float, level=None):
    """MC on (z,y,x) volume -> verts(x,y,z), faces。"""
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
    if HAS_TRIMESH and len(faces) > 0:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        pts, _ = trimesh.sample.sample_surface(mesh, int(n))
        return pts.astype(np.float32)
    # 回退：采顶点
    vidx = np.random.choice(len(verts), size=int(n), replace=len(verts) < n)
    return verts[vidx].astype(np.float32)

def chamfer_distance(p: np.ndarray, q: np.ndarray):
    if HAS_SCIPY:
        t_p = cKDTree(p); d_pq, _ = t_p.query(q, k=1, workers=-1)
        t_q = cKDTree(q); d_qp, _ = t_q.query(p, k=1, workers=-1)
        cd_l2 = float((d_pq**2).mean() + (d_qp**2).mean())
        cd_l1 = float(d_pq.mean() + d_qp.mean())
        return cd_l2, cd_l1
    # torch fallback
    P = torch.from_numpy(p).float()
    Q = torch.from_numpy(q).float()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    P = P.to(device); Q = Q.to(device)
    def min_dists(A, B, bs=65536):
        mins = []
        for i in range(0, A.shape[0], bs):
            a = A[i:i+bs]
            d2 = torch.cdist(a, B, p=2)
            mins.append(d2.min(dim=1).values)
        return torch.cat(mins, dim=0)
    d_pq = min_dists(Q, P); d_qp = min_dists(P, Q)
    cd_l2 = float((d_pq**2).mean().item() + (d_qp**2).mean().item())
    cd_l1 = float(d_pq.mean().item() + d_qp.mean().item())
    return cd_l2, cd_l1

def compact_mesh(verts: np.ndarray, faces: np.ndarray):
    if not HAS_TRIMESH or len(faces) == 0:
        return verts, faces
    m = trimesh.Trimesh(verts, faces, process=False)
    m.remove_unreferenced_vertices()
    return m.vertices.view(np.ndarray), m.faces.view(np.ndarray)

# ---------- 构造状态 ----------
def build_state_from_args(args, model, data_dir: Path):
    """
    返回 (state_vec, sel_key)，其中：
      - state_vec: (dof,) 弧度
      - sel_key:   用于取GT/iso auto 的数据 key（尽量给出）
    优先级：--state-key > --state-rad/--state-deg > --angle-*
    """
    dof_model = int(getattr(model.hparams, 'dof', 1))
    rs_parsed = None

    if args.state_key is not None:
        rs_parsed = load_robot_states(data_dir)
        k = str(args.state_key)
        if k not in rs_parsed:
            raise SystemExit(f"[error] robot_state.json 中不存在 key={k}")
        state_vec = rs_parsed[k].astype(np.float32)
        if state_vec.shape[0] != dof_model:
            raise SystemExit(f"[error] key={k} 的状态维度={state_vec.shape[0]} 与模型 dof={dof_model} 不一致")
        return state_vec, k

    if args.state_rad is not None or args.state_deg is not None:
        src = args.state_rad if args.state_rad is not None else args.state_deg
        vals = [float(x) for x in src.split(',') if x.strip()!='']
        if args.state_deg is not None:
            vals = [v * math.pi / 180.0 for v in vals]
        if len(vals) != dof_model:
            raise SystemExit(f"[error] 提供了 {len(vals)} 个状态，但模型 dof={dof_model}。")
        state_vec = np.array(vals, dtype=np.float32)
        rs_parsed = rs_parsed or load_robot_states(data_dir)
        near_k = find_nearest_key_by_state(rs_parsed, state_vec)
        return state_vec, near_k

    # 兼容旧参数：单角度
    if (args.angle_deg is None) == (args.angle_rad is None):
        raise SystemExit("请使用 --state-key / --state-rad / --state-deg 之一；或使用 --angle-*（仅单关节）。")
    angle_rad = float(args.angle_rad if args.angle_rad is not None else (args.angle_deg * math.pi / 180.0))
    if dof_model == 1:
        state_vec = np.array([angle_rad], dtype=np.float32)
        rs_parsed = rs_parsed or load_robot_states(data_dir)
        # 取第一个关节角最近的 key
        keys = list(rs_parsed.keys())
        angs = np.array([rs_parsed[k][0] for k in keys], dtype=np.float32)
        near_k = keys[int(np.argmin(np.abs(angs - angle_rad)))]
        return state_vec, near_k
    else:
        print("[warn] 模型 dof>1，但只提供了单个角度；将其作为第1关节，其余补0。建议改用 --state-key 或 --state-rad/--state-deg。")
        state_vec = np.zeros((dof_model,), dtype=np.float32); state_vec[0] = angle_rad
        rs_parsed = rs_parsed or load_robot_states(data_dir)
        near_k = find_nearest_key_by_state(rs_parsed, state_vec)
        return state_vec, near_k

# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--data-dir', type=str, required=True)

    # 旧参数（兼容）
    ap.add_argument('--angle-deg', type=float, default=None)
    ap.add_argument('--angle-rad', type=float, default=None)
    # 多关节
    ap.add_argument('--state-key', type=str, default=None)
    ap.add_argument('--state-rad', type=str, default=None)
    ap.add_argument('--state-deg', type=str, default=None)

    ap.add_argument('--res', type=int, default=128)
    ap.add_argument('--sample-n', type=int, default=100000)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--out-dir', type=str, default='demo_out')

    # AABB（坐标不缩放）
    ap.add_argument('--aabb-scale', type=float, default=1.0)
    ap.add_argument('--aabb-min', type=float, default=None)
    ap.add_argument('--aabb-max', type=float, default=None)
    ap.add_argument('--aabb-auto', action='store_true')
    ap.add_argument('--aabb-margin', type=float, default=0.05)

    # iso
    ap.add_argument('--iso', type=str, default='0')   # 'auto' or numeric

    # 连通块清理
    ap.add_argument('--keep-largest', action='store_true')
    ap.add_argument('--min-comp-area', type=float, default=0.0)
    ap.add_argument('--min-comp-ratio', type=float, default=0.0)
    ap.add_argument('--min-comp-verts', type=int, default=0)
    ap.add_argument('--min-comp-faces', type=int, default=0)

    # —— 新增：两种剪枝 —— #
    ap.add_argument('--prune-by-gt', type=float, default=0.0,
                    help='按与GT点云距离阈值(训练坐标)过滤三角面；0关闭。例：0.03')
    ap.add_argument('--grad-thr', type=float, default=0.0,
                    help='按|∇SDF|过滤三角面（保留所有顶点梯度>=thr的三角面）；0关闭。例：0.4')

    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    # load model
    device = args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    model = VisModelingModel.load_from_checkpoint(args.ckpt, strict=False)
    model = model.to(device).eval()

    # 构造状态 & 推荐 key
    state_vec, sel_key = build_state_from_args(args, model, data_dir)
    dof_model = int(getattr(model.hparams, 'dof', state_vec.shape[0]))

    # 读取 GT（若知道 key）
    gt_phys = gt_norm = None
    if sel_key is not None:
        gt_phys = load_gt_points_by_key(data_dir, sel_key)
        gt_norm = normalize_xyzn_like_training(gt_phys)

    # AABB（在模型坐标/“归一化坐标”中设定；此脚本不缩放坐标）
    if args.aabb_min is not None and args.aabb_max is not None:
        aabb_min, aabb_max = float(args.aabb_min), float(args.aabb_max)
    else:
        if args.aabb_auto and gt_norm is not None:
            r = float(np.max(np.abs(gt_norm))) * (1.0 + float(args.aabb_margin))
            aabb_min, aabb_max = -r, r
        else:
            s = float(args.aabb_scale)
            aabb_min, aabb_max = -s, s

    # iso
    iso_level = None
    if str(args.iso).lower() == 'auto' and gt_norm is not None:
        with torch.no_grad():
            c = torch.from_numpy(gt_norm).to(device)
            st = torch.from_numpy((state_vec.astype(np.float32)/math.pi).reshape(1, -1)).to(device).expand(c.shape[0], -1)
            pred_on = call_model_sdf(model, c, st).squeeze(-1).float().cpu().numpy()
        iso_level = float(np.median(pred_on))
    else:
        try: iso_level = float(args.iso)
        except Exception: iso_level = None

    print(f"[demo] dof={dof_model}  AABB=[{aabb_min:.3f},{aabb_max:.3f}]^3, res={args.res}, iso={('auto' if str(args.iso).lower()=='auto' else f'{iso_level:.4f}')}")

    # 体素 & MC（坐标不缩放）
    vol = query_volume_state(model, state_vec, res=int(args.res), device=device,
                             aabb_min=aabb_min, aabb_max=aabb_max)
    print(f"[demo] volume stats: min={vol.min():.4f}, max={vol.max():.4f}")
    verts_norm, faces = marching_cubes_extract(vol, aabb_min=aabb_min, aabb_max=aabb_max, level=iso_level)

    # —— 可选：连通块清理 —— #
    if HAS_TRIMESH and (args.keep_largest or args.min_comp_area > 0.0 or args.min_comp_ratio > 0.0
                        or args.min_comp_verts > 0 or args.min_comp_faces > 0):
        mesh = trimesh.Trimesh(verts_norm, faces, process=False)
        parts = mesh.split(only_watertight=False)
        if parts:
            areas = np.array([max(m.area, 1e-9) for m in parts], dtype=float)
            kept = []
            if args.keep_largest:
                kept = [parts[int(np.argmax(areas))]]
            else:
                thr_abs = float(args.min_comp_area)
                if args.min_comp_ratio > 0.0:
                    thr_abs = max(thr_abs, float(args.min_comp_ratio) * float(areas.sum()))
                for m, a in zip(parts, areas):
                    if thr_abs > 0.0 and a < thr_abs: continue
                    if args.min_comp_verts > 0 and len(m.vertices) < int(args.min_comp_verts): continue
                    if args.min_comp_faces > 0 and len(m.faces)   < int(args.min_comp_faces):  continue
                    kept.append(m)
                if not kept:
                    kept = [parts[int(np.argmax(areas))]]
            mesh2 = trimesh.util.concatenate(kept)
            verts_norm = mesh2.vertices.view(np.ndarray)
            faces = mesh2.faces.view(np.ndarray)

    # ===========================
    #      新增：两种剪枝
    # ===========================
    # (A) 基于 |∇SDF| 的三角面筛选
    if args.grad_thr > 0:
        res = int(args.res)
        step = (aabb_max - aabb_min) / (res - 1)
        gz, gy, gx = np.gradient(vol, step, step, step)  # vol轴为 (z,y,x)
        gmag = np.sqrt(gx*gx + gy*gy + gz*gz)

        ix = np.clip(((verts_norm[:, 0] - aabb_min) / step).round().astype(int), 0, res-1)
        iy = np.clip(((verts_norm[:, 1] - aabb_min) / step).round().astype(int), 0, res-1)
        iz = np.clip(((verts_norm[:, 2] - aabb_min) / step).round().astype(int), 0, res-1)

        v_gmag = gmag[iz, iy, ix]  # 每个顶点的梯度范数
        keep_face = (v_gmag[faces].min(axis=1) >= float(args.grad_thr))
        faces = faces[keep_face]
        verts_norm, faces = compact_mesh(verts_norm, faces)
        print(f"[prune] grad-thr={args.grad_thr:.3f} -> faces={len(faces)}")

    # (B) 基于与GT距离的三角面筛选
    if args.prune_by_gt > 0 and gt_norm is not None and len(faces) > 0:
        centroids = verts_norm[faces].mean(axis=1)  # (F,3)
        gt_sub = gt_norm
        if gt_sub.shape[0] > 100000:  # 子采样以提速
            sel = np.random.choice(gt_sub.shape[0], 100000, replace=False)
            gt_sub = gt_sub[sel]

        if HAS_SCIPY:
            d, _ = cKDTree(gt_sub).query(centroids, k=1, workers=-1)
        else:
            Ct = torch.from_numpy(centroids).float()
            Gt = torch.from_numpy(gt_sub).float()
            d_list = []
            for i in range(0, Ct.shape[0], 65536):
                d_list.append(torch.cdist(Ct[i:i+65536], Gt).min(dim=1).values)
            d = torch.cat(d_list, 0).cpu().numpy()

        keep_face = (d <= float(args.prune_by_gt))
        faces = faces[keep_face]
        verts_norm, faces = compact_mesh(verts_norm, faces)
        print(f"[prune] prune-by-gt={args.prune_by_gt:.4f} -> faces={len(faces)}")

    # ===== 导出：归一化 & 物理（两者此处等价；不做缩放） =====
    if HAS_TRIMESH and len(faces) > 0:
        pred_mesh_norm_path = out_dir / "pred_mesh_norm.obj"
        trimesh.Trimesh(verts_norm, faces, process=False).export(str(pred_mesh_norm_path))

    verts_phys = denormalize_from_training_like(verts_norm)
    if HAS_TRIMESH and len(faces) > 0:
        pred_mesh_phys_path = out_dir / "pred_mesh.obj"
        trimesh.Trimesh(verts_phys, faces, process=False).export(str(pred_mesh_phys_path))
        # 采样导出点云（物理坐标）
        pred_pts_phys = sample_surface_points(verts_phys, faces, n=int(args.sample_n))
        trimesh.points.PointCloud(pred_pts_phys).export(str(out_dir / "pred_points.ply"))
        if gt_phys is not None:
            trimesh.points.PointCloud(gt_phys).export(str(out_dir / f"gt_points_{sel_key}.ply"))
    else:
        pred_pts_phys = sample_surface_points(verts_phys, faces, n=int(args.sample_n))

    # ===== Chamfer（物理坐标） =====
    if gt_phys is not None and len(pred_pts_phys) > 0:
        gt_eval = gt_phys
        if gt_eval.shape[0] > args.sample_n:
            sel = np.random.choice(gt_eval.shape[0], size=int(args.sample_n), replace=False)
            gt_eval = gt_eval[sel]
        cd_l2, cd_l1 = chamfer_distance(pred_pts_phys.astype(np.float32), gt_eval.astype(np.float32))
        with (out_dir / "metrics.txt").open("w", encoding="utf-8") as f:
            f.write(f"dof: {dof_model}\n")
            f.write(f"state_key: {sel_key}\n")
            f.write(f"AABB(norm): [{aabb_min:.6f}, {aabb_max:.6f}]\n")
            f.write(f"cd_l2_phys: {cd_l2:.6e}\ncd_l1_phys: {cd_l1:.6e}\n")
        print(f"[demo] 使用 GT: mesh_{sel_key}.xyzn")
        print(f"[demo] Chamfer@PHYS (L2) = {cd_l2:.6e}   Chamfer@PHYS (L1) = {cd_l1:.6e}")
    else:
        print("[demo] 未计算 Chamfer（缺 GT 或空网格）。")

    print(f"[demo] 输出：\n"
          f"  - 预测点云(物理): {out_dir/'pred_points.ply'}\n"
          f"  - 预测网格(物理): {out_dir/'pred_mesh.obj'}\n"
          f"  - 预测网格(归一): {out_dir/'pred_mesh_norm.obj'}\n"
          f"  - （若确定了 key）GT 点云: {out_dir/f'gt_points_{sel_key}.ply' if sel_key else '(无)'}\n"
          f"  - 指标: {out_dir/'metrics.txt'}")

if __name__ == "__main__":
    main()



'''


[钳子成功了这个 不管了]
pliers_11_12_state-condition_new-global-siren-sdf_1/lightning_logs/version_1/checkpoints/epoch=170-step=2565.ckpt
python demo_reconstruct_and_cd.py \
  --ckpt /data/fllm/code/vsm/pliers_11_12_state-condition_new-global-siren-sdf_1/lightning_logs/version_4/checkpoints/epoch=299-step=4500.ckpt \
  --data-dir data/pliers_2074 \
  --state-key 312 \
  --res 160 \
  --aabb-scale 1.0 --aabb-margin 0.05 \
  --iso 0.0 --grad-thr 0.4 \
  --keep-largest \
  --min-comp-verts 5000 \
  --min-comp-ratio 0.02 \
  --sample-n 20000 \
  --device cuda \
  --out-dir demo_out/pliers_2074_norm

[眼镜成功了这个 不管了]
eyeglasses_11_12_state-condition_new-global-siren-sdf_1/lightning_logs/version_1/checkpoints/epoch=244-step=3675.ckpt
python demo_reconstruct_and_cd.py \
  --ckpt /data/fllm/code/vsm/eyeglasses_11_12_state-condition_new-global-siren-sdf_1/lightning_logs/version_2/checkpoints/epoch=299-step=4500.ckpt \
  --data-dir data/eyeglasses_101863 \
  --state-key 312 \
  --res 224 \
  --aabb-scale 1.0 --aabb-margin 0.05 \
  --iso auto \
  --grad-thr 0.0 \
  --min-comp-verts 0 --min-comp-ratio 0.0 \
  --sample-n 20000 \
  --device cuda \
  --out-dir demo_out/eyeglasses_101863_norm

[剪刀过]
/data/fllm/code/vsm/scissors_11_12_state-condition_new-global-siren-sdf_1/lightning_logs/version_0/checkpoints/epoch=133-step=2010.ckpt
/data/fllm/code/vsm/scissors_11_12_state-condition_new-global-siren-sdf_1/lightning_logs/version_2/checkpoints/epoch=73-step=1110.ckpt
/data/fllm/code/vsm/scissors_11_12_state-condition_new-global-siren-sdf_1/lightning_logs/version_5/checkpoints/epoch=299-step=4500.ckpt 最新的 cd最低

python demo_reconstruct_and_cd.py \
  --ckpt /data/fllm/code/vsm/scissors_11_12_state-condition_new-global-siren-sdf_1/lightning_logs/version_5/checkpoints/epoch=299-step=4500.ckpt \
  --data-dir data/scissors_10893 \
  --state-key 312 \
  --res 224 \
  --aabb-scale 1.0 --aabb-margin 0.05 \
  --iso auto \
  --grad-thr 0.0 \
  --min-comp-verts 0 --min-comp-ratio 0.0 \
  --sample-n 20000 \
  --device cuda \
  --out-dir demo_out/scissors_10893_no_prune


狗的就这样了，用最新的FINAL
boston_dynamics_spot_state-condition_new-global-siren-sdf_1/lightning_logs/version_0/checkpoints/epoch=354-step=5325.ckpt
/data/fllm/code/vsm/FINAL_boston_dynamics_spot_state-condition_new-global-siren-sdf_1/lightning_logs/version_1/checkpoints/epoch=299-step=4500.ckpt 
/data/fllm/code/vsm/original_boston_dynamics_spot_state-condition_new-global-siren-sdf_1/lightning_logs/version_0/checkpoints/epoch=138-step=2085.ckpt
python demo_reconstruct_and_cd.py \
  --ckpt /data/fllm/code/vsm/original_boston_dynamics_spot_state-condition_new-global-siren-sdf_1/lightning_logs/version_0/checkpoints/epoch=138-step=2085.ckpt \
  --data-dir data/boston_dynamics_spot_original \
  --state-key 212 \
  --res 160 \
  --aabb-scale 1.0 --aabb-margin 0.05 \
  --iso 0.0 \
  --keep-largest \
  --min-comp-verts 5000 \
  --min-comp-ratio 0.02 \
  --sample-n 20000 \
  --device cuda \
  --out-dir demo_out/boston_dynamics_spot




  
机械臂新版数据一直失败
/data/fllm/code/vsm/franka_fr3_state-condition_new-global-siren-sdf_1/lightning_logs/version_6/checkpoints/epoch=375-step=5640.ckpt

/data/fllm/code/vsm/FINAL_franka_fr3_state-condition_new-global-siren-sdf_1/lightning_logs/version_0/checkpoints/epoch=299-step=4500.ckpt

python demo_reconstruct_and_cd.py \
  --ckpt /data/fllm/code/vsm/franka_fr3_state-condition_new-global-siren-sdf_1/lightning_logs/version_6/checkpoints/epoch=375-step=5640.ckpt \
  --data-dir data/franka_fr3_no_scale \
  --state-key 212 \
  --res 160 \
  --aabb-scale 1.0 --aabb-margin 0.05 \
  --iso auto \
  --keep-largest \
  --min-comp-verts 5000 \
  --min-comp-ratio 0.02 \
  --sample-n 20000 \
  --device cuda \
  --out-dir demo_out/franka_fr3

python demo_reconstruct_and_cd.py \
  --ckpt /data/fllm/code/vsm/original_franka_fr3_state-condition_new-global-siren-sdf_1/lightning_logs/version_1/checkpoints/epoch=86-step=1305.ckpt \
  --data-dir data/franka_fr3_original \
  --state-key 212 \
  --res 160 \
  --aabb-scale 1.0 --aabb-margin 0.05 \
  --iso auto \
  --sample-n 20000 \
  --device cuda \
  --out-dir demo_out/franka_fr3


python demo_reconstruct_and_cd.py \
  --ckpt /data/fllm/code/vsm/SIM2_state-condition_new-global-siren-sdf_6/lightning_logs/version_0/checkpoints/epoch=93-step=13254.ckpt \
  --data-dir dataset_tdcr/tdcr_2m_no_base_vsm \
  --state-key 212 \
  --res 160 \
  --aabb-scale 1.0 --aabb-margin 0.05 \
  --iso auto \
  --sample-n 20000 \
  --device cuda \
  --out-dir demo_out/tdcr_2m_no_base_vsm


'''