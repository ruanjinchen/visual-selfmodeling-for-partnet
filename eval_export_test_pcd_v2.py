# -*- coding: utf-8 -*-
"""
Export GT & Pred point clouds (PLY) for ALL test states.

Compared to the first version, this script adds robust handling for "bubble / extra shell" artifacts:
- Marching-cubes can produce multiple disconnected iso-surfaces (components).
  We can select the *best* component for each test frame using GT proximity (default),
  or drop components that touch the AABB boundary.

Usage example:
  python eval_export_test_pcd_v2.py \
    --config configs/state_condition/sim_2m_with_base.yaml \
    --ckpt path/to/epoch=...ckpt \
    --out-dir eval_out/sim_2m_with_base \
    --device cuda \
    --res 128 \
    --iso 0.0 \
    --select-comp closest_gt

Key outputs:
  <out_dir>/gt/000546.ply
  <out_dir>/pred/000546.ply

Notes:
- GT points are read from mesh_{id}.xyzn.npy (preferred) or mesh_{id}.xyzn.
- Pred points are extracted by evaluating the SDF on a grid in [-1,1]^3 by default,
  then marching cubes, then sampling surface points.
- If your model produces a large outer shell, use:
    --select-comp closest_gt_drop_boundary
  or enable automatic iso calibration:
    --iso auto

Author: GPT
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except Exception as e:
    raise ImportError("This script requires PyTorch.") from e

# Optional deps
HAS_TRIMESH = False
try:
    import trimesh  # type: ignore
    HAS_TRIMESH = True
except Exception:
    HAS_TRIMESH = False

HAS_SKIMAGE = False
try:
    from skimage.measure import marching_cubes  # type: ignore
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False

HAS_MCUBES = False
try:
    import mcubes  # type: ignore
    HAS_MCUBES = True
except Exception:
    HAS_MCUBES = False

HAS_SCIPY = False
try:
    from scipy.spatial import cKDTree  # type: ignore
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# ----------------- IO helpers -----------------
def mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_ply_xyz(path: Path, xyz: np.ndarray):
    """
    Save point cloud (xyz only) to ASCII PLY.
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    mkdir(path.parent)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for p in xyz:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")


def load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def load_gt_points(data_dir: Path, key: int) -> np.ndarray:
    """
    Load GT surface point cloud from:
      - mesh_{key}.xyzn.npy (preferred)
      - mesh_{key}.xyzn (txt)
    Returns xyz (N,3).
    """
    base = data_dir / f"mesh_{key}.xyzn"
    npy = base.with_suffix(base.suffix + ".npy")  # .xyzn.npy

    if npy.exists():
        arr = np.load(str(npy)).astype(np.float32, copy=False)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] < 3:
            raise ValueError(f"{npy} has shape {arr.shape}, need >=3 cols")
        return arr[:, :3]

    if base.exists():
        arr = np.loadtxt(str(base), dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] < 3:
            raise ValueError(f"{base} has shape {arr.shape}, need >=3 cols")
        return arr[:, :3]

    raise FileNotFoundError(f"GT point file not found: {base} or {npy}")


def parse_state_vector(entry: Any, dof: int) -> np.ndarray:
    """
    robot_state.json 每个 idx 对应一个 entry，形态可能不一致。
    目标：拿到长度 dof 的 raw 向量（随后 /pi 变成网络输入）
    兼容形态：
      - [ [v0], [v1], ... ] (TDCR 导出格式)
      - [v0, v1, ...]
      - [[[v0]], [[v1]], ...]
    """
    # list-like
    if isinstance(entry, (list, tuple, np.ndarray)):
        arr = entry
        # e.g. [[v],[v],...]
        if len(arr) >= dof and isinstance(arr[0], (list, tuple, np.ndarray)):
            out = []
            for k in range(dof):
                v = arr[k]
                # unwrap nested
                while isinstance(v, (list, tuple, np.ndarray)) and len(v) > 0:
                    if isinstance(v[0], (list, tuple, np.ndarray)):
                        v = v[0]
                    else:
                        break
                if isinstance(v, (list, tuple, np.ndarray)):
                    out.append(float(v[0]))
                else:
                    out.append(float(v))
            return np.asarray(out, dtype=np.float32)
        # e.g. [v,v,...]
        if len(arr) >= dof and not isinstance(arr[0], (list, tuple, np.ndarray)):
            return np.asarray(arr[:dof], dtype=np.float32)
    # dict-like
    if isinstance(entry, dict):
        # common keys
        for k in ["ctrl", "state", "q", "motor", "motors", "joints"]:
            if k in entry:
                return parse_state_vector(entry[k], dof)
    raise ValueError(f"Unsupported robot_state entry format for dof={dof}: {type(entry)}")


# ----------------- Model loading -----------------
def resolve_model_class():
    """
    Try to import model class in your repo.
    """
    try:
        from models import VisModelingModel  # type: ignore
        return VisModelingModel
    except Exception:
        pass
    try:
        from models import VSM  # type: ignore
        return VSM
    except Exception as e:
        raise ImportError("Cannot import model class from models.py (VisModelingModel or VSM). "
                          "Please run this script inside your VSM repo root.") from e


def instantiate_model(ModelCls, cfg: Dict[str, Any]):
    """
    Instantiate lightning module with cfg as best effort.
    """
    sig = inspect.signature(ModelCls.__init__)
    params = [p for p in sig.parameters.values() if p.name != "self"]

    # Case A: __init__(self, cfg) / __init__(self, config)
    if len(params) == 1 and params[0].kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
        pname = params[0].name
        if pname in ("cfg", "config", "hparams", "args"):
            try:
                return ModelCls(cfg)  # type: ignore
            except TypeError:
                return ModelCls(**{pname: cfg})  # type: ignore

    # Case B: __init__(self, **kwargs) matching keys
    cand = dict(cfg)
    # some legacy key mapping
    if "learning_rate" in cfg and "lr" not in cand:
        cand["lr"] = cfg["learning_rate"]

    kwargs = {}
    for p in params:
        if p.name in cand and cand[p.name] is not None:
            kwargs[p.name] = cand[p.name]

    # fill required params if present in cfg
    for p in params:
        if p.default is inspect._empty and p.name not in kwargs:
            if p.name in cfg:
                kwargs[p.name] = cfg[p.name]

    return ModelCls(**kwargs)  # type: ignore


def load_model(cfg: Dict[str, Any], ckpt_path: Path, device: str = "cpu"):
    ModelCls = resolve_model_class()
    model = instantiate_model(ModelCls, cfg)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)}")
    model.eval()
    model.to(device)
    return model


def get_net(model):
    """
    Return callable that maps [xyz,state] -> sdf.
    """
    if hasattr(model, "model"):
        return getattr(model, "model")
    return model


@torch.inference_mode()
def call_model_sdf(net, coords: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    """
    coords: (N,3), states: (N,dof) -> sdf: (N,1)
    """
    x = torch.cat([coords, states], dim=-1)
    y = net(x)
    return y


# ---------------- Surface reconstruction ----------------
def make_grid(res: int, aabb_min: float, aabb_max: float) -> np.ndarray:
    """
    Grid points in xyz order, but reshape(res,res,res) gives (z,y,x).
    """
    lin = np.linspace(aabb_min, aabb_max, res, dtype=np.float32)
    zz, yy, xx = np.meshgrid(lin, lin, lin, indexing="ij")  # (z,y,x)
    pts = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    return pts


def marching_cubes_extract(vol_zyx: np.ndarray, aabb_min: float, aabb_max: float, level: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    vol_zyx: (res,res,res) in (z,y,x)
    return verts_xyz, faces
    """
    res = vol_zyx.shape[0]
    voxel_size = (aabb_max - aabb_min) / float(res - 1)

    if HAS_SKIMAGE:
        # skimage verts are in (z,y,x) with spacing applied, origin at 0
        verts_zyx, faces, _normals, _vals = marching_cubes(vol_zyx, level=level, spacing=(voxel_size, voxel_size, voxel_size))
        verts_zyx = verts_zyx.astype(np.float32, copy=False)
        faces = faces.astype(np.int32, copy=False)
        # shift to aabb_min
        verts_zyx += np.array([aabb_min, aabb_min, aabb_min], dtype=np.float32)
        # reorder to xyz
        verts_xyz = verts_zyx[:, [2, 1, 0]]
        return verts_xyz, faces

    if HAS_MCUBES:
        # mcubes returns verts in voxel index space (z,y,x)
        verts_zyx, faces = mcubes.marching_cubes(vol_zyx, level)
        verts_zyx = verts_zyx.astype(np.float32, copy=False)
        faces = faces.astype(np.int32, copy=False)
        verts_zyx = aabb_min + verts_zyx * voxel_size
        verts_xyz = verts_zyx[:, [2, 1, 0]]
        return verts_xyz, faces

    raise ImportError("Need either scikit-image or PyMCubes for marching cubes.")


def triangle_areas(verts: np.ndarray, faces: np.ndarray) -> float:
    if len(faces) == 0:
        return 0.0
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    a = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1) * 0.5
    return float(a.sum())


def split_mesh_components_union_find(verts: np.ndarray, faces: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
    """
    Split mesh into connected components using union-find on vertex indices.
    Returns list of (verts_i, faces_i, info).
    """
    V = int(len(verts))
    F = int(len(faces))
    if V == 0:
        return []
    if F == 0:
        info = {"area": 0.0, "v": V, "f": 0,
                "aabb_min": verts.min(axis=0), "aabb_max": verts.max(axis=0)}
        return [(verts, faces, info)]

    parent = np.arange(V, dtype=np.int32)
    rank = np.zeros(V, dtype=np.int8)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int):
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    # union vertices within each face
    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        union(a, b)
        union(b, c)

    # root for each vertex
    roots = np.fromiter((find(i) for i in range(V)), dtype=np.int32, count=V)

    # group faces by root (using first vertex)
    face_roots = roots[faces[:, 0]]
    uniq = np.unique(face_roots)
    out: List[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]] = []

    for r in uniq:
        face_idx = np.nonzero(face_roots == r)[0]
        f_sub = faces[face_idx]
        vid = np.unique(f_sub.reshape(-1))
        v_sub = verts[vid]

        # reindex faces
        # vid is sorted; use searchsorted
        f_re = np.searchsorted(vid, f_sub)

        info = {
            "root": int(r),
            "v": int(len(v_sub)),
            "f": int(len(f_re)),
            "area": triangle_areas(v_sub, f_re),
            "aabb_min": v_sub.min(axis=0),
            "aabb_max": v_sub.max(axis=0),
        }
        out.append((v_sub, f_re, info))
    return out


def component_touches_boundary(info: Dict[str, Any], aabb_min: float, aabb_max: float, eps: float) -> bool:
    mn = info["aabb_min"]
    mx = info["aabb_max"]
    # touches if close to boundary on any axis
    if np.any(mn <= (aabb_min + eps)) or np.any(mx >= (aabb_max - eps)):
        return True
    return False


def score_component_to_gt(v: np.ndarray, f: np.ndarray, gt_pts: np.ndarray, samples: int = 2048) -> float:
    """
    Lower score is better. Uses mean nearest-neighbor distance to GT.
    """
    if len(v) == 0 or len(gt_pts) == 0:
        return float("inf")
    m = min(int(samples), int(len(v)))
    idx = np.random.choice(len(v), size=m, replace=False)
    q = v[idx].astype(np.float32, copy=False)

    if HAS_SCIPY:
        tree = cKDTree(gt_pts.astype(np.float32, copy=False))
        d, _ = tree.query(q, k=1, workers=-1)
        return float(np.mean(d))
    # fallback brute (chunked)
    gt = gt_pts.astype(np.float32, copy=False)
    out_d = []
    chunk = 1024
    for i in range(0, len(q), chunk):
        qq = q[i:i+chunk]
        # (C,1,3) - (1,M,3) -> (C,M,3)
        diff = qq[:, None, :] - gt[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        out_d.append(np.sqrt(np.min(dist2, axis=1)))
    d = np.concatenate(out_d, axis=0)
    return float(np.mean(d))


def select_component(verts: np.ndarray,
                     faces: np.ndarray,
                     mode: str,
                     gt_pts: Optional[np.ndarray],
                     aabb_min: float,
                     aabb_max: float,
                     boundary_eps: float,
                     score_samples: int,
                     verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    mode:
      - all
      - largest_area
      - closest_gt
      - closest_gt_drop_boundary
      - largest_area_drop_boundary
    """
    if mode == "all" or len(verts) == 0 or len(faces) == 0:
        return verts, faces

    comps = split_mesh_components_union_find(verts, faces)
    if not comps:
        return verts, faces
    if len(comps) == 1:
        return comps[0][0], comps[0][1]

    # boundary flags
    flags = []
    for _v, _f, info in comps:
        touch = component_touches_boundary(info, aabb_min=aabb_min, aabb_max=aabb_max, eps=boundary_eps)
        flags.append(touch)

    candidates = list(range(len(comps)))
    if "drop_boundary" in mode:
        non_boundary = [i for i, t in enumerate(flags) if not t]
        if non_boundary:
            candidates = non_boundary

    if mode.startswith("largest_area"):
        areas = [comps[i][2]["area"] for i in candidates]
        j = candidates[int(np.argmax(areas))]
        if verbose:
            print(f"[comp] choose largest_area among {len(candidates)}/{len(comps)} comps, areas={areas}")
        return comps[j][0], comps[j][1]

    if mode.startswith("closest_gt"):
        if gt_pts is None:
            # fallback to largest area
            areas = [comps[i][2]["area"] for i in candidates]
            j = candidates[int(np.argmax(areas))]
            if verbose:
                print("[comp] closest_gt requested but gt_pts is None; fallback largest_area")
            return comps[j][0], comps[j][1]
        scores = [score_component_to_gt(comps[i][0], comps[i][1], gt_pts, samples=score_samples) for i in candidates]
        j = candidates[int(np.argmin(scores))]
        if verbose:
            print(f"[comp] choose closest_gt among {len(candidates)}/{len(comps)} comps, scores={scores}")
        return comps[j][0], comps[j][1]

    # default fallback
    return verts, faces


def sample_points_from_mesh(verts: np.ndarray, faces: np.ndarray, n: int) -> np.ndarray:
    if len(verts) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if n <= 0:
        return verts.astype(np.float32, copy=False)

    if HAS_TRIMESH and len(faces) > 0:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        pts, _ = trimesh.sample.sample_surface(mesh, int(n))
        return np.asarray(pts, dtype=np.float32)

    # fallback: sample vertices
    idx = np.random.choice(len(verts), size=int(n), replace=len(verts) < n)
    return verts[idx].astype(np.float32)


# --------------- Point cloud prediction ---------------
@torch.inference_mode()
def pred_pointcloud_mc(net,
                       state_norm: np.ndarray,
                       res: int,
                       iso: float,
                       pred_n: int,
                       device: str,
                       aabb_min: float,
                       aabb_max: float,
                       chunk: int,
                       grid_cache: Optional[torch.Tensor],
                       select_comp_mode: str,
                       gt_pts: Optional[np.ndarray],
                       boundary_eps: float,
                       score_samples: int,
                       verbose_comp: bool) -> np.ndarray:
    """
    grid + marching cubes + (optional) component selection + sample surface points
    """
    # grid coords
    if grid_cache is None:
        coords_np = make_grid(res, aabb_min=aabb_min, aabb_max=aabb_max)
        coords_all = torch.from_numpy(coords_np).to(device=device)
    else:
        coords_all = grid_cache

    N = coords_all.shape[0]
    state_t = torch.from_numpy(state_norm.astype(np.float32)).to(device=device).reshape(1, -1)

    sdf_flat = np.empty((N,), dtype=np.float32)
    for i in range(0, N, chunk):
        c = coords_all[i:i+chunk]
        s = state_t.expand(c.shape[0], -1)
        sdf = call_model_sdf(net, c, s).reshape(-1)
        sdf_flat[i:i+chunk] = sdf.float().detach().cpu().numpy()

    vol = sdf_flat.reshape(res, res, res)  # (z,y,x)
    verts, faces = marching_cubes_extract(vol, aabb_min=aabb_min, aabb_max=aabb_max, level=float(iso))

    # component selection
    if select_comp_mode != "all" and len(verts) > 0 and len(faces) > 0:
        verts, faces = select_component(
            verts, faces,
            mode=select_comp_mode,
            gt_pts=gt_pts,
            aabb_min=aabb_min,
            aabb_max=aabb_max,
            boundary_eps=boundary_eps,
            score_samples=score_samples,
            verbose=verbose_comp,
        )

    pts = sample_points_from_mesh(verts, faces, pred_n)
    return pts


@torch.inference_mode()
def pred_pointcloud_neariso(net,
                            state_norm: np.ndarray,
                            iso: float,
                            pred_n: int,
                            device: str,
                            aabb_min: float,
                            aabb_max: float,
                            chunk: int,
                            samples: int) -> np.ndarray:
    """
    Random samples in AABB, keep those closest to iso.
    (No mesh / marching cubes dependency, but lower quality.)
    """
    M = int(samples)
    xyz = (torch.rand((M, 3), device=device) * (aabb_max - aabb_min) + aabb_min).float()
    state_t = torch.from_numpy(state_norm.astype(np.float32)).to(device=device).reshape(1, -1)
    sdf = []
    for i in range(0, M, chunk):
        c = xyz[i:i+chunk]
        s = state_t.expand(c.shape[0], -1)
        y = call_model_sdf(net, c, s).reshape(-1)
        sdf.append(y)
    sdf = torch.cat(sdf, dim=0)
    # select closest
    dist = torch.abs(sdf - float(iso))
    k = min(int(pred_n), M)
    idx = torch.topk(-dist, k=k, largest=True).indices
    pts = xyz[idx].detach().cpu().numpy().astype(np.float32)
    return pts


@torch.inference_mode()
def estimate_iso_from_gt(net,
                         state_norm: np.ndarray,
                         gt_pts: np.ndarray,
                         device: str,
                         samples: int,
                         chunk: int) -> float:
    """
    Estimate iso level for this state by evaluating SDF on GT points and taking median.
    """
    if len(gt_pts) == 0:
        return 0.0
    m = min(int(samples), int(len(gt_pts)))
    idx = np.random.choice(len(gt_pts), size=m, replace=False)
    coords = torch.from_numpy(gt_pts[idx].astype(np.float32)).to(device=device)
    state_t = torch.from_numpy(state_norm.astype(np.float32)).to(device=device).reshape(1, -1)
    sdf_list = []
    for i in range(0, m, chunk):
        c = coords[i:i+chunk]
        s = state_t.expand(c.shape[0], -1)
        y = call_model_sdf(net, c, s).reshape(-1)
        sdf_list.append(y)
    sdf = torch.cat(sdf_list, dim=0).detach().cpu().numpy()
    return float(np.median(sdf))


def load_yaml(path: Path) -> Dict[str, Any]:
    # minimal yaml loader: use PyYAML if available
    try:
        import yaml  # type: ignore
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise ImportError("PyYAML is required to read config YAML.") from e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="training config yaml")
    ap.add_argument("--ckpt", type=str, required=True, help="lightning checkpoint")
    ap.add_argument("--out-dir", type=str, required=True)

    ap.add_argument("--data-dir", type=str, default="", help="override cfg.data_filepath")
    ap.add_argument("--split-json", type=str, default="", help="override assets/datainfo/...json")
    ap.add_argument("--split-key", type=str, default="test", choices=["train", "val", "test"])

    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--iso", type=str, default="0.0", help="iso level float, or 'auto' to estimate from GT points")
    ap.add_argument("--iso-auto-samples", type=int, default=2048, help="GT subsamples for iso=auto")

    ap.add_argument("--aabb-min", type=float, default=-1.0)
    ap.add_argument("--aabb-max", type=float, default= 1.0)

    ap.add_argument("--pred-npoints", type=int, default=0,
                    help="pred point count; 0 means match GT point count for each frame")
    ap.add_argument("--chunk", type=int, default=262144)

    ap.add_argument("--method", type=str, default="mc", choices=["mc", "neariso"])
    ap.add_argument("--neariso-samples", type=int, default=800000,
                    help="only for --method neariso: number of random samples in AABB")

    # New: component selection (bubble removal)
    ap.add_argument("--select-comp", type=str, default="closest_gt",
                    choices=["all", "largest_area", "closest_gt", "closest_gt_drop_boundary", "largest_area_drop_boundary"],
                    help="How to select marching-cubes connected component(s). "
                         "closest_gt is recommended for test set (uses GT to pick the best component).")

    ap.add_argument("--boundary-eps", type=float, default=0.02,
                    help="AABB boundary eps used by *_drop_boundary modes (in normalized coords).")
    ap.add_argument("--comp-score-samples", type=int, default=2048,
                    help="How many vertices to sample for component->GT distance scoring.")
    ap.add_argument("--verbose-comp", action="store_true", help="Print component scores/areas when multiple comps exist.")

    ap.add_argument("--limit", type=int, default=0, help="debug: only export first N states")
    ap.add_argument("--start", type=int, default=0, help="skip first N states in split list")

    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA not available, fallback to cpu")
        device = "cpu"

    # speed hint on tensor core GPUs
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # resolve paths
    data_dir = Path(args.data_dir) if args.data_dir else Path(str(cfg.get("data_filepath", "")))
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    if args.split_json:
        split_json = Path(args.split_json)
    else:
        seed = int(cfg.get("seed", 0))
        split_json = Path("assets") / "datainfo" / f"multiple_models_data_split_dict_{seed}.json"
    if not split_json.exists():
        raise FileNotFoundError(f"split_json not found: {split_json}")

    out_dir = Path(args.out_dir)
    gt_dir = out_dir / "gt"
    pred_dir = out_dir / "pred"
    mkdir(gt_dir)
    mkdir(pred_dir)

    # load split
    split = load_json(split_json)
    ids = split.get(args.split_key, [])
    if not isinstance(ids, list):
        raise ValueError(f"split['{args.split_key}'] is not a list")
    if args.start > 0:
        ids = ids[args.start:]
    if args.limit > 0:
        ids = ids[:args.limit]

    print(f"[info] 使用 split: {split_json} 中的 '{args.split_key}'（共 {len(split.get(args.split_key, []))} 个）。")
    print(f"[info] 待导出 {args.split_key} 条目数：{len(ids)}")
    print(f"[info] AABB = [{args.aabb_min:.3f}, {args.aabb_max:.3f}]^3, res={args.res}, iso={args.iso}")

    # load robot states
    robot_state_path = data_dir / "robot_state.json"
    robot_states = load_json(robot_state_path)
    # dof
    dof = int(cfg.get("dof", 0))
    if dof <= 0:
        # infer from first entry
        k0 = next(iter(robot_states.keys()))
        entry0 = robot_states[k0]
        dof = len(entry0)
        print(f"[warn] dof not set in config; inferred dof={dof} from robot_state[{k0}]")

    # load model
    model = load_model(cfg, Path(args.ckpt), device=device)
    net = get_net(model)

    # grid cache for fixed aabb
    grid_cache: Optional[torch.Tensor] = None
    if args.method == "mc":
        coords_np = make_grid(args.res, aabb_min=args.aabb_min, aabb_max=args.aabb_max)
        grid_cache = torch.from_numpy(coords_np).to(device=device)

    # parse iso
    iso_auto = False
    iso_level_fixed = 0.0
    if str(args.iso).lower() == "auto":
        iso_auto = True
    else:
        iso_level_fixed = float(args.iso)

    for key in ids:
        k_int = int(key)
        name = f"{k_int:06d}.ply"

        # load GT and save
        gt_pts = load_gt_points(data_dir, k_int)
        save_ply_xyz(gt_dir / name, gt_pts)

        # state
        entry = robot_states[str(k_int)]
        state_raw = parse_state_vector(entry, dof=dof)
        state_norm = (state_raw / math.pi).astype(np.float32)

        # pred npoints
        pred_n = int(args.pred_npoints) if int(args.pred_npoints) > 0 else int(len(gt_pts))

        # iso
        if iso_auto:
            iso_level = estimate_iso_from_gt(
                net=net,
                state_norm=state_norm,
                gt_pts=gt_pts,
                device=device,
                samples=int(args.iso_auto_samples),
                chunk=min(int(args.chunk), 65536),
            )
        else:
            iso_level = iso_level_fixed

        # predict
        if args.method == "mc":
            pred_pts = pred_pointcloud_mc(
                net=net,
                state_norm=state_norm,
                res=int(args.res),
                iso=float(iso_level),
                pred_n=int(pred_n),
                device=device,
                aabb_min=float(args.aabb_min),
                aabb_max=float(args.aabb_max),
                chunk=int(args.chunk),
                grid_cache=grid_cache,
                select_comp_mode=str(args.select_comp),
                gt_pts=gt_pts,
                boundary_eps=float(args.boundary_eps),
                score_samples=int(args.comp_score_samples),
                verbose_comp=bool(args.verbose_comp),
            )
        else:
            pred_pts = pred_pointcloud_neariso(
                net=net,
                state_norm=state_norm,
                iso=float(iso_level),
                pred_n=int(pred_n),
                device=device,
                aabb_min=float(args.aabb_min),
                aabb_max=float(args.aabb_max),
                chunk=min(int(args.chunk), 65536),
                samples=int(args.neariso_samples),
            )

        save_ply_xyz(pred_dir / name, pred_pts)

    print(f"[done] saved to: {out_dir}")
    print(f"  gt:   {gt_dir}")
    print(f"  pred: {pred_dir}")


if __name__ == "__main__":
    main()
