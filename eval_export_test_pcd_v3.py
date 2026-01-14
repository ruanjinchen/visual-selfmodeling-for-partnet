# -*- coding: utf-8 -*-
"""eval_export_test_pcd_v3.py

Export GT & Pred point clouds (PLY) for ALL test states.

This version adds **GT-free** ways to suppress the "big bubble / outer shell" artifact.

Why the bubble happens:
  - Marching-cubes can produce multiple disconnected iso-surfaces.
  - A large outer shell often has much larger area than the true robot surface.
  - If we sample points from *all* triangles (or keep the largest component), the shell dominates.

GT-free solutions implemented here:
  1) drop components that touch the AABB boundary (common for the outer shell)
  2) pick the component closest to the origin or to user-provided seed points
  3) optionally shrink the AABB during extraction

If you DO have GT (offline evaluation), you can still use closest_gt / iso=auto.

Typical real-test usage (NO GT):
  python eval_export_test_pcd_v3.py \
    --config configs/state_condition/sim_2m_with_base.yaml \
    --ckpt path/to/ckpt.ckpt \
    --out-dir out/sim_2m_with_base \
    --device cuda \
    --res 128 \
    --iso 0.0 \
    --no-gt \
    --pred-npoints 20000 \
    --select-comp closest_origin_drop_boundary \
    --boundary-eps 0.03

Author: GPT
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
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


def mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_ply_xyz(path: Path, xyz: np.ndarray):
    """Save xyz point cloud to ASCII PLY."""
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


def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore

        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise ImportError("PyYAML is required to read config YAML.") from e


def load_gt_points(data_dir: Path, key: int) -> np.ndarray:
    """Load GT surface xyz from mesh_{key}.xyzn(.npy)."""
    base = data_dir / f"mesh_{key}.xyzn"
    npy = base.with_suffix(base.suffix + ".npy")  # .xyzn.npy
    if npy.exists():
        arr = np.load(str(npy)).astype(np.float32, copy=False)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr[:, :3]
    if base.exists():
        arr = np.loadtxt(str(base), dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr[:, :3]
    raise FileNotFoundError(f"GT point file not found: {base} or {npy}")


def parse_state_vector(entry: Any, dof: int) -> np.ndarray:
    """Parse one robot_state.json entry into raw (dof,) float32."""
    if isinstance(entry, (list, tuple, np.ndarray)):
        arr = entry
        # [[v],[v],...]
        if len(arr) >= dof and isinstance(arr[0], (list, tuple, np.ndarray)):
            out = []
            for k in range(dof):
                v = arr[k]
                while isinstance(v, (list, tuple, np.ndarray)) and len(v) > 0 and isinstance(v[0], (list, tuple, np.ndarray)):
                    v = v[0]
                if isinstance(v, (list, tuple, np.ndarray)):
                    out.append(float(v[0]))
                else:
                    out.append(float(v))
            return np.asarray(out, dtype=np.float32)
        # [v,v,...]
        if len(arr) >= dof and not isinstance(arr[0], (list, tuple, np.ndarray)):
            return np.asarray(arr[:dof], dtype=np.float32)
    if isinstance(entry, dict):
        for k in ["ctrl", "state", "q", "motor", "motors", "joints"]:
            if k in entry:
                return parse_state_vector(entry[k], dof)
    raise ValueError(f"Unsupported robot_state entry format for dof={dof}: {type(entry)}")


def resolve_model_class():
    try:
        from models import VisModelingModel  # type: ignore

        return VisModelingModel
    except Exception:
        pass
    try:
        from models import VSM  # type: ignore

        return VSM
    except Exception as e:
        raise ImportError(
            "Cannot import model class from models.py (VisModelingModel or VSM). "
            "Please run this script inside your VSM repo root."
        ) from e


def instantiate_model(ModelCls, cfg: Dict[str, Any]):
    sig = inspect.signature(ModelCls.__init__)
    params = [p for p in sig.parameters.values() if p.name != "self"]

    # __init__(cfg)
    if len(params) == 1:
        pname = params[0].name
        if pname in ("cfg", "config", "hparams", "args"):
            try:
                return ModelCls(cfg)  # type: ignore
            except TypeError:
                return ModelCls(**{pname: cfg})  # type: ignore

    cand = dict(cfg)
    if "learning_rate" in cfg and "lr" not in cand:
        cand["lr"] = cfg["learning_rate"]

    kwargs = {}
    for p in params:
        if p.name in cand and cand[p.name] is not None:
            kwargs[p.name] = cand[p.name]
    for p in params:
        if p.default is inspect._empty and p.name not in kwargs and p.name in cfg:
            kwargs[p.name] = cfg[p.name]

    return ModelCls(**kwargs)  # type: ignore


def load_model(cfg: Dict[str, Any], ckpt_path: Path, device: str):
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
    return getattr(model, "model") if hasattr(model, "model") else model


@torch.inference_mode()
def call_model_sdf(net, coords: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    x = torch.cat([coords, states], dim=-1)
    return net(x)


def make_grid(res: int, aabb_min: float, aabb_max: float) -> np.ndarray:
    lin = np.linspace(aabb_min, aabb_max, res, dtype=np.float32)
    zz, yy, xx = np.meshgrid(lin, lin, lin, indexing="ij")
    return np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)


def marching_cubes_extract(vol_zyx: np.ndarray, aabb_min: float, aabb_max: float, level: float) -> Tuple[np.ndarray, np.ndarray]:
    res = vol_zyx.shape[0]
    voxel = (aabb_max - aabb_min) / float(res - 1)
    if HAS_SKIMAGE:
        verts_zyx, faces, _n, _v = marching_cubes(vol_zyx, level=level, spacing=(voxel, voxel, voxel))
        verts_zyx = verts_zyx.astype(np.float32, copy=False)
        faces = faces.astype(np.int32, copy=False)
        verts_zyx += np.array([aabb_min, aabb_min, aabb_min], dtype=np.float32)
        verts_xyz = verts_zyx[:, [2, 1, 0]]
        return verts_xyz, faces
    if HAS_MCUBES:
        verts_zyx, faces = mcubes.marching_cubes(vol_zyx, level)
        verts_zyx = verts_zyx.astype(np.float32, copy=False)
        faces = faces.astype(np.int32, copy=False)
        verts_zyx = aabb_min + verts_zyx * voxel
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
    V = int(len(verts))
    if V == 0:
        return []
    F = int(len(faces))
    if F == 0:
        info = {
            "area": 0.0,
            "v": V,
            "f": 0,
            "aabb_min": verts.min(axis=0),
            "aabb_max": verts.max(axis=0),
        }
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

    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        union(a, b)
        union(b, c)

    roots = np.fromiter((find(i) for i in range(V)), dtype=np.int32, count=V)
    face_roots = roots[faces[:, 0]]
    uniq = np.unique(face_roots)
    out: List[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]] = []
    for r in uniq:
        face_idx = np.nonzero(face_roots == r)[0]
        f_sub = faces[face_idx]
        vid = np.unique(f_sub.reshape(-1))
        v_sub = verts[vid]
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
    return bool(np.any(mn <= (aabb_min + eps)) or np.any(mx >= (aabb_max - eps)))


def score_component_to_gt(v: np.ndarray, gt_pts: np.ndarray, samples: int = 2048) -> float:
    if len(v) == 0 or len(gt_pts) == 0:
        return float("inf")
    m = min(int(samples), int(len(v)))
    idx = np.random.choice(len(v), size=m, replace=False)
    q = v[idx].astype(np.float32, copy=False)
    if HAS_SCIPY:
        tree = cKDTree(gt_pts.astype(np.float32, copy=False))
        d, _ = tree.query(q, k=1, workers=-1)
        return float(np.mean(d))
    # brute fallback
    diff = q[:, None, :] - gt_pts[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    return float(np.mean(np.sqrt(np.min(dist2, axis=1))))


def score_component_to_origin(v: np.ndarray) -> float:
    """Lower is better: use median radius to origin."""
    if len(v) == 0:
        return float("inf")
    r = np.linalg.norm(v.astype(np.float32, copy=False), axis=1)
    return float(np.median(r))


def parse_seed_points(s: str) -> np.ndarray:
    """Parse 'x,y,z; x,y,z; ...' into (K,3)."""
    s = s.strip()
    if not s:
        return np.zeros((0, 3), dtype=np.float32)
    pts = []
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        nums = [float(x) for x in part.split(",")]
        if len(nums) != 3:
            raise ValueError(f"Bad seed point '{part}', need x,y,z")
        pts.append(nums)
    return np.asarray(pts, dtype=np.float32)


def score_component_to_seeds(v: np.ndarray, seeds: np.ndarray, samples: int = 4096) -> float:
    """Lower is better: mean distance from each seed to nearest vertex."""
    if len(v) == 0 or len(seeds) == 0:
        return float("inf")
    v = v.astype(np.float32, copy=False)
    seeds = seeds.astype(np.float32, copy=False)
    if HAS_SCIPY:
        tree = cKDTree(v)
        d, _ = tree.query(seeds, k=1, workers=-1)
        return float(np.mean(d))
    # fallback: sample vertices
    m = min(int(samples), int(len(v)))
    idx = np.random.choice(len(v), size=m, replace=False)
    vv = v[idx]
    d_all = []
    for p in seeds:
        dist2 = np.sum((vv - p[None, :]) ** 2, axis=1)
        d_all.append(math.sqrt(float(np.min(dist2))))
    return float(np.mean(d_all))


def select_component(
    verts: np.ndarray,
    faces: np.ndarray,
    mode: str,
    gt_pts: Optional[np.ndarray],
    aabb_min: float,
    aabb_max: float,
    boundary_eps: float,
    score_samples: int,
    seed_points: Optional[np.ndarray],
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Select ONE mesh component."""
    if mode == "all" or len(verts) == 0 or len(faces) == 0:
        return verts, faces
    comps = split_mesh_components_union_find(verts, faces)
    if not comps:
        return verts, faces
    if len(comps) == 1:
        return comps[0][0], comps[0][1]

    touches = [component_touches_boundary(info, aabb_min, aabb_max, boundary_eps) for _v, _f, info in comps]
    cand = list(range(len(comps)))
    if mode.endswith("_drop_boundary"):
        nb = [i for i, t in enumerate(touches) if not t]
        if nb:
            cand = nb

    def pick_by(scores: List[float], kind: str) -> int:
        j = cand[int(np.argmin(scores))]
        if verbose:
            print(f"[comp] {kind}: cand={len(cand)}/{len(comps)}, scores={scores}")
        return j

    if mode.startswith("largest_area"):
        areas = [comps[i][2]["area"] for i in cand]
        j = cand[int(np.argmax(areas))]
        if verbose:
            print(f"[comp] largest_area: cand={len(cand)}/{len(comps)}, areas={areas}")
        return comps[j][0], comps[j][1]

    if mode.startswith("closest_gt"):
        if gt_pts is None:
            # fallback
            areas = [comps[i][2]["area"] for i in cand]
            j = cand[int(np.argmax(areas))]
            if verbose:
                print("[comp] closest_gt requested but no gt; fallback largest_area")
            return comps[j][0], comps[j][1]
        scores = [score_component_to_gt(comps[i][0], gt_pts, samples=score_samples) for i in cand]
        j = pick_by(scores, "closest_gt")
        return comps[j][0], comps[j][1]

    if mode.startswith("closest_origin"):
        scores = [score_component_to_origin(comps[i][0]) for i in cand]
        j = pick_by(scores, "closest_origin")
        return comps[j][0], comps[j][1]

    if mode.startswith("closest_seeds"):
        if seed_points is None or len(seed_points) == 0:
            raise ValueError("closest_seeds requires --seed-points")
        scores = [score_component_to_seeds(comps[i][0], seed_points, samples=max(score_samples, 4096)) for i in cand]
        j = pick_by(scores, "closest_seeds")
        return comps[j][0], comps[j][1]

    # default
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
    idx = np.random.choice(len(verts), size=int(n), replace=len(verts) < n)
    return verts[idx].astype(np.float32)


@torch.inference_mode()
def pred_pointcloud_mc(
    net,
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
    seed_points: Optional[np.ndarray],
    verbose_comp: bool,
) -> np.ndarray:
    if grid_cache is None:
        coords_np = make_grid(res, aabb_min, aabb_max)
        coords_all = torch.from_numpy(coords_np).to(device=device)
    else:
        coords_all = grid_cache

    N = coords_all.shape[0]
    state_t = torch.from_numpy(state_norm.astype(np.float32)).to(device=device).reshape(1, -1)
    sdf_flat = np.empty((N,), dtype=np.float32)
    for i in range(0, N, chunk):
        c = coords_all[i : i + chunk]
        s = state_t.expand(c.shape[0], -1)
        y = call_model_sdf(net, c, s).reshape(-1)
        sdf_flat[i : i + chunk] = y.float().detach().cpu().numpy()

    vol = sdf_flat.reshape(res, res, res)  # (z,y,x)
    verts, faces = marching_cubes_extract(vol, aabb_min, aabb_max, level=float(iso))

    if select_comp_mode != "all" and len(verts) > 0 and len(faces) > 0:
        verts, faces = select_component(
            verts,
            faces,
            mode=select_comp_mode,
            gt_pts=gt_pts,
            aabb_min=aabb_min,
            aabb_max=aabb_max,
            boundary_eps=boundary_eps,
            score_samples=score_samples,
            seed_points=seed_points,
            verbose=verbose_comp,
        )

    return sample_points_from_mesh(verts, faces, pred_n)


@torch.inference_mode()
def estimate_iso_from_gt(net, state_norm: np.ndarray, gt_pts: np.ndarray, device: str, samples: int, chunk: int) -> float:
    if len(gt_pts) == 0:
        return 0.0
    m = min(int(samples), int(len(gt_pts)))
    idx = np.random.choice(len(gt_pts), size=m, replace=False)
    coords = torch.from_numpy(gt_pts[idx].astype(np.float32)).to(device=device)
    state_t = torch.from_numpy(state_norm.astype(np.float32)).to(device=device).reshape(1, -1)
    sdf_list = []
    for i in range(0, m, chunk):
        c = coords[i : i + chunk]
        s = state_t.expand(c.shape[0], -1)
        y = call_model_sdf(net, c, s).reshape(-1)
        sdf_list.append(y)
    sdf = torch.cat(sdf_list, dim=0).detach().cpu().numpy()
    return float(np.median(sdf))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)

    ap.add_argument("--data-dir", type=str, default="", help="override cfg.data_filepath")
    ap.add_argument("--split-json", type=str, default="", help="override assets/datainfo/...json")
    ap.add_argument("--split-key", type=str, default="test", choices=["train", "val", "test"])

    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--iso", type=str, default="0.0", help="float or 'auto'(needs GT)")
    ap.add_argument("--iso-auto-samples", type=int, default=2048)

    ap.add_argument("--aabb-min", type=float, default=-1.0)
    ap.add_argument("--aabb-max", type=float, default=1.0)

    ap.add_argument("--pred-npoints", type=int, default=20000)
    ap.add_argument("--chunk", type=int, default=262144)

    ap.add_argument("--no-gt", action="store_true", help="Do not read/save GT (for real test without scans).")

    ap.add_argument(
        "--select-comp",
        type=str,
        default="closest_origin_drop_boundary",
        choices=[
            "all",
            "largest_area",
            "largest_area_drop_boundary",
            "closest_origin",
            "closest_origin_drop_boundary",
            "closest_seeds",
            "closest_seeds_drop_boundary",
            "closest_gt",
            "closest_gt_drop_boundary",
        ],
    )
    ap.add_argument("--boundary-eps", type=float, default=0.02)
    ap.add_argument("--comp-score-samples", type=int, default=2048)
    ap.add_argument("--seed-points", type=str, default="0,0,0", help="For closest_seeds*: 'x,y,z; x,y,z; ...'")
    ap.add_argument("--verbose-comp", action="store_true")

    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)

    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA not available, fallback to cpu")
        device = "cpu"
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

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
    mkdir(pred_dir)
    if not args.no_gt:
        mkdir(gt_dir)

    split = load_json(split_json)
    ids = split.get(args.split_key, [])
    if not isinstance(ids, list):
        raise ValueError(f"split['{args.split_key}'] is not a list")
    if args.start > 0:
        ids = ids[args.start :]
    if args.limit > 0:
        ids = ids[: args.limit]

    print(f"[info] split={split_json} key={args.split_key} n={len(ids)}")
    print(f"[info] AABB=[{args.aabb_min:.3f},{args.aabb_max:.3f}] res={args.res} iso={args.iso} select={args.select_comp}")

    robot_states = load_json(data_dir / "robot_state.json")
    dof = int(cfg.get("dof", 0))
    if dof <= 0:
        k0 = next(iter(robot_states.keys()))
        dof = len(robot_states[k0])
        print(f"[warn] dof not set; inferred dof={dof} from robot_state[{k0}]")

    model = load_model(cfg, Path(args.ckpt), device=device)
    net = get_net(model)

    # grid cache
    coords_np = make_grid(int(args.res), float(args.aabb_min), float(args.aabb_max))
    grid_cache = torch.from_numpy(coords_np).to(device=device)

    iso_auto = str(args.iso).lower() == "auto"
    if iso_auto and args.no_gt:
        raise ValueError("--iso auto requires GT points, but --no-gt is set")
    iso_fixed = 0.0 if iso_auto else float(args.iso)

    seed_points = parse_seed_points(args.seed_points) if args.select_comp.startswith("closest_seeds") else None

    for key in ids:
        k_int = int(key)
        name = f"{k_int:06d}.ply"

        gt_pts: Optional[np.ndarray] = None
        if not args.no_gt:
            gt_pts = load_gt_points(data_dir, k_int)
            save_ply_xyz(gt_dir / name, gt_pts)

        entry = robot_states[str(k_int)]
        state_raw = parse_state_vector(entry, dof=dof)
        state_norm = (state_raw / math.pi).astype(np.float32)

        if iso_auto:
            assert gt_pts is not None
            iso_level = estimate_iso_from_gt(net, state_norm, gt_pts, device=device, samples=int(args.iso_auto_samples), chunk=min(int(args.chunk), 65536))
        else:
            iso_level = iso_fixed

        pred_pts = pred_pointcloud_mc(
            net=net,
            state_norm=state_norm,
            res=int(args.res),
            iso=float(iso_level),
            pred_n=int(args.pred_npoints if args.no_gt else (len(gt_pts) if gt_pts is not None else args.pred_npoints)),
            device=device,
            aabb_min=float(args.aabb_min),
            aabb_max=float(args.aabb_max),
            chunk=int(args.chunk),
            grid_cache=grid_cache,
            select_comp_mode=str(args.select_comp),
            gt_pts=gt_pts,
            boundary_eps=float(args.boundary_eps),
            score_samples=int(args.comp_score_samples),
            seed_points=seed_points,
            verbose_comp=bool(args.verbose_comp),
        )
        save_ply_xyz(pred_dir / name, pred_pts)

    print(f"[done] saved to: {out_dir}")


if __name__ == "__main__":
    main()
'''

python eval_export_test_pcd_v3.py \
  --config configs/state_condition/sim_2m_with_base.yaml \
  --ckpt sim_2m_with_base_state-condition_new-global-siren-sdf_6/lightning_logs/version_0/checkpoints/epoch=499-step=70500.ckpt \
  --out-dir eval_out/sim_2m_with_base_predonly \
  --device cuda \
  --res 128 \
  --iso 0.0 \
  --no-gt \
  --pred-npoints 20000 \
  --select-comp closest_origin_drop_boundary --aabb-min -0.8 --aabb-max 0.8 \
  --boundary-eps 0.03

'''