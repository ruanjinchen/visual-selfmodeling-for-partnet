# -*- coding: utf-8 -*-
"""
Export GT & Pred point clouds (PLY) for ALL test states.

需求对应：
- 读取 split JSON 中的 test 列表（序号 idx）
- 从 data_dir/robot_state.json 读取该 idx 的电机/关节状态
- 用训练好的 VSM (state-conditioned SDF) 在该状态下生成 *点云*（不保存 mesh）
- 同时读取 GT 点云（mesh_{idx}.xyzn(.npy) 的前三列）并保存
- 输出：
    <out_dir>/gt/000546.ply
    <out_dir>/pred/000546.ply
  文件名以 test idx 为准，补齐 6 位

说明：
- 默认在训练坐标系（AABB=[-1,1]^3）工作与导出；不做反归一化。
- 预测点云生成默认用 marching cubes 提取 iso-surface 后，从表面采样点（质量最好）。
  若环境缺少 skimage / PyMCubes，可用 --method neariso 作为无依赖替代（质量较差）。
- 本脚本借鉴官方 repo eval.py 的“按状态查询 SDF 并重建”的流程。

用法示例：
python eval_export_test_pcd.py \
  --config configs/state_condition/sim_5m_with_base.yaml \
  --ckpt /path/to/epoch=499-step=xxxx.ckpt \
  --out-dir eval_pcd/sim_5m_no_base \
  --res 128 \
  --device cuda

可选：
  --split-json assets/datainfo/multiple_models_data_split_dict_43.json
  --data-dir dataset_tdcr/tdcr_5m_no_base_vsm
  --pred-npoints 20000
  --method mc | neariso
"""

from __future__ import annotations

import os
import json
import math
import argparse
import inspect
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm

import torch
import yaml

# --- optional deps ---
try:
    import trimesh  # type: ignore
    HAS_TRIMESH = True
except Exception:
    HAS_TRIMESH = False

try:
    from skimage.measure import marching_cubes  # type: ignore
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False

try:
    import mcubes  # type: ignore
    HAS_MCUBES = True
except Exception:
    HAS_MCUBES = False


# ---------------- IO utils ----------------
def mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_ply_xyz(path: Path, xyz: np.ndarray) -> None:
    """
    Save point cloud (xyz) to PLY (binary_little_endian).
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be (N,3), got {xyz.shape}")

    mkdir(path.parent)
    with open(path, "wb") as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {xyz.shape[0]}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "end_header\n"
        )
        f.write(header.encode("ascii"))
        f.write(xyz.astype("<f4").tobytes())


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Config YAML should be a dict, got {type(obj)}")
    return obj


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
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
    robot_state.json 每个 idx 对应一个 entry，形态在不同工程里可能不一致。
    目标：拿到长度 dof 的 raw 向量（随后 /pi 变成网络输入）
    兼容常见形态：
      - [ [v0], [v1], ... ] (你的 TDCR 导出格式)
      - [v0, v1, ...]
      - [[[v0]], [[v1]], ...]（官方某些数据）
    """
    # 先尝试“按维度列表”的结构
    if isinstance(entry, list) and len(entry) >= dof:
        vec: List[float] = []
        for i in range(dof):
            v = entry[i]
            # v could be scalar or nested list
            while isinstance(v, list) and len(v) > 0:
                v = v[0]
            vec.append(float(v))
        return np.asarray(vec, dtype=np.float32)

    # fallback: 深度优先 flatten
    flat: List[float] = []

    def _dfs(x: Any) -> None:
        if isinstance(x, list):
            for y in x:
                _dfs(y)
        else:
            try:
                flat.append(float(x))
            except Exception:
                pass

    _dfs(entry)
    if len(flat) < dof:
        raise ValueError(f"State entry cannot provide dof={dof} values, got {len(flat)}. entry type={type(entry)}")
    return np.asarray(flat[:dof], dtype=np.float32)


# ---------------- Model loading ----------------
def resolve_model_class():
    """
    尝试兼容两种常见命名：
      - models.VisModelingModel（官方）
      - models.VSM（你修改后的版本可能用这个）
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
    尽量用 cfg 自动匹配 __init__ 参数并实例化。
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

    # Case B: many kwargs like official VisModelingModel
    log_dir = "_".join([str(cfg.get("log_dir", "log")),
                        str(cfg.get("model_name", "model")),
                        str(cfg.get("tag", "tag")),
                        str(cfg.get("seed", "0"))])

    cand = dict(
        lr=cfg.get("lr", 1e-4),
        seed=cfg.get("seed", 0),
        dof=cfg.get("dof", None),
        if_cuda=cfg.get("if_cuda", True),
        if_test=True,
        gamma=cfg.get("gamma", 0.5),
        log_dir=log_dir,
        train_batch=cfg.get("train_batch", 1),
        val_batch=cfg.get("val_batch", 1),
        test_batch=cfg.get("test_batch", 1),
        task_batch=cfg.get("task_batch", 1),
        num_workers=cfg.get("num_workers", 0),
        model_name=cfg.get("model_name", "state-condition"),
        data_filepath=cfg.get("data_filepath", cfg.get("data_dir", "")),
        lr_schedule=cfg.get("lr_schedule", []),
        num_gpus=cfg.get("num_gpus", 1),
        epochs=cfg.get("epochs", 1),
        loss_type=cfg.get("loss_type", "siren_sdf"),
        coord_system=cfg.get("coord_system", "cartesian"),
        tag=cfg.get("tag", ""),
        cache_to=cfg.get("cache_to", "cpu"),
    )

    # only pass supported params
    kwargs = {}
    for p in params:
        if p.name in cand and cand[p.name] is not None:
            kwargs[p.name] = cand[p.name]

    # sanity: fill required params that are missing but present in cfg
    for p in params:
        if p.default is inspect._empty and p.name not in kwargs:
            if p.name in cfg:
                kwargs[p.name] = cfg[p.name]

    return ModelCls(**kwargs)  # type: ignore


def load_model(cfg: Dict[str, Any], ckpt_path: Path, device: str):
    ModelCls = resolve_model_class()
    model = instantiate_model(ModelCls, cfg)

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] Missing keys when loading ckpt (show up to 20): {missing[:20]}")
    if unexpected:
        print(f"[warn] Unexpected keys when loading ckpt (show up to 20): {unexpected[:20]}")

    model = model.to(device)
    model.eval()
    # freeze if LightningModule provides
    if hasattr(model, "freeze"):
        try:
            model.freeze()
        except Exception:
            pass
    return model


def get_net(model):
    """
    Return the callable network that maps [xyz,state] -> sdf.
    """
    if hasattr(model, "model"):
        return getattr(model, "model")
    return model


@torch.inference_mode()
def call_model_sdf(net, coords: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    """
    coords: (M,3), states: (M,dof) already normalized (divide by pi)
    return: (M,1) or (M,)
    """
    x = torch.cat([coords, states], dim=-1)
    y = net(x)
    return y


# ---------------- Surface reconstruction ----------------
def make_grid(res: int, aabb_min: float, aabb_max: float) -> np.ndarray:
    """
    Return grid points in xyz order, but arranged such that reshape(res,res,res) gives (z,y,x).
    """
    lin = np.linspace(aabb_min, aabb_max, res, dtype=np.float32)
    zz, yy, xx = np.meshgrid(lin, lin, lin, indexing="ij")  # (z,y,x)
    pts = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)  # (N,3) xyz
    return pts


def marching_cubes_extract(vol_zyx: np.ndarray, aabb_min: float, aabb_max: float, level: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    vol_zyx: (res,res,res) in (z,y,x)
    return verts_xyz, faces
    """
    res = vol_zyx.shape[0]
    voxel_size = (aabb_max - aabb_min) / float(res - 1)

    if HAS_SKIMAGE:
        # skimage returns verts in (z,y,x) coordinates with spacing applied, origin at 0
        verts_zyx, faces, _normals, _vals = marching_cubes(vol_zyx, level=level, spacing=(voxel_size, voxel_size, voxel_size))
        verts_zyx = verts_zyx.astype(np.float32, copy=False)
        faces = faces.astype(np.int32, copy=False)
        # convert to xyz + add aabb_min offset
        verts_xyz = np.empty_like(verts_zyx)
        verts_xyz[:, 0] = verts_zyx[:, 2] + aabb_min
        verts_xyz[:, 1] = verts_zyx[:, 1] + aabb_min
        verts_xyz[:, 2] = verts_zyx[:, 0] + aabb_min
        return verts_xyz, faces

    if HAS_MCUBES:
        # mcubes returns verts in voxel index space (z,y,x)
        verts_zyx, faces = mcubes.marching_cubes(vol_zyx, level)
        verts_zyx = np.asarray(verts_zyx, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int32)
        verts_xyz = np.empty_like(verts_zyx)
        verts_xyz[:, 0] = verts_zyx[:, 2] * voxel_size + aabb_min
        verts_xyz[:, 1] = verts_zyx[:, 1] * voxel_size + aabb_min
        verts_xyz[:, 2] = verts_zyx[:, 0] * voxel_size + aabb_min
        return verts_xyz, faces

    raise ImportError("Need skimage or PyMCubes for marching cubes. "
                      "Install one of: `pip install scikit-image` or `pip install PyMCubes`.")


def maybe_keep_largest_component(verts: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if not HAS_TRIMESH or len(verts) == 0 or len(faces) == 0:
        return verts, faces
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    parts = mesh.split(only_watertight=False)
    if not parts:
        return verts, faces
    # pick largest area
    areas = np.array([max(p.area, 1e-9) for p in parts], dtype=float)
    m = parts[int(np.argmax(areas))]
    return m.vertices.view(np.ndarray), m.faces.view(np.ndarray)


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
                       keep_largest: bool) -> np.ndarray:
    """
    grid + marching cubes + sample surface points
    """
    # prepare grid coords
    if grid_cache is None:
        pts = make_grid(res, aabb_min=aabb_min, aabb_max=aabb_max)  # (N,3)
        coords_all = torch.from_numpy(pts).to(device)
    else:
        coords_all = grid_cache

    N = coords_all.shape[0]
    dof = int(state_norm.shape[0])
    state_t = torch.from_numpy(state_norm.astype(np.float32)).to(device).view(1, dof)

    sdf_flat = np.empty((N,), dtype=np.float32)
    for i in range(0, N, chunk):
        c = coords_all[i:i+chunk]
        s = state_t.expand(c.shape[0], -1)
        sdf = call_model_sdf(net, c, s).reshape(-1)
        sdf_flat[i:i+chunk] = sdf.float().detach().cpu().numpy()

    vol = sdf_flat.reshape(res, res, res)  # (z,y,x)
    verts, faces = marching_cubes_extract(vol, aabb_min=aabb_min, aabb_max=aabb_max, level=float(iso))
    if keep_largest:
        verts, faces = maybe_keep_largest_component(verts, faces)
    pts = sample_points_from_mesh(verts, faces, pred_n)
    return pts


@torch.inference_mode()
def pred_pointcloud_neariso(net,
                            state_norm: np.ndarray,
                            pred_n: int,
                            device: str,
                            aabb_min: float,
                            aabb_max: float,
                            n_samples: int,
                            chunk: int,
                            iso: float) -> np.ndarray:
    """
    Uniform random samples in AABB, pick points with smallest |sdf-iso|.
    No marching cubes dependency, but quality worse.
    """
    dof = int(state_norm.shape[0])
    state_t1 = torch.from_numpy(state_norm.astype(np.float32)).to(device).view(1, dof)

    # sample on CPU then send in chunks
    pts = (np.random.rand(int(n_samples), 3).astype(np.float32) * (aabb_max - aabb_min) + aabb_min)
    coords_all = torch.from_numpy(pts).to(device)
    sdf_all = np.empty((pts.shape[0],), dtype=np.float32)

    for i in range(0, pts.shape[0], chunk):
        c = coords_all[i:i+chunk]
        s = state_t1.expand(c.shape[0], -1)
        sdf = call_model_sdf(net, c, s).reshape(-1)
        sdf_all[i:i+chunk] = sdf.float().detach().cpu().numpy()

    dist = np.abs(sdf_all - float(iso))
    k = min(int(pred_n), dist.shape[0])
    idx = np.argpartition(dist, kth=k-1)[:k]
    return pts[idx].astype(np.float32, copy=False)


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="training yaml config")
    ap.add_argument("--ckpt", type=str, required=True, help="Lightning .ckpt")
    ap.add_argument("--out-dir", type=str, required=True, help="output root; will create gt/ and pred/")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--split-json", type=str, default=None, help="override split json path; default uses seed from config")
    ap.add_argument("--data-dir", type=str, default=None, help="override dataset dir; default uses data_filepath from config")

    ap.add_argument("--res", type=int, default=128, help="grid resolution for marching cubes")
    ap.add_argument("--iso", type=float, default=0.0)
    ap.add_argument("--aabb-min", type=float, default=-1.0)
    ap.add_argument("--aabb-max", type=float, default= 1.0)

    ap.add_argument("--pred-npoints", type=int, default=0,
                    help="pred point count; 0 means match GT point count for each frame")
    ap.add_argument("--chunk", type=int, default=262144)

    ap.add_argument("--method", type=str, default="mc", choices=["mc", "neariso"])
    ap.add_argument("--neariso-samples", type=int, default=800000,
                    help="only for --method neariso: number of random samples in AABB")

    ap.add_argument("--keep-largest", action="store_true", help="keep largest component (needs trimesh)")

    ap.add_argument("--limit", type=int, default=0, help="debug: only export first N test states")
    ap.add_argument("--start", type=int, default=0, help="debug: skip first N test states")

    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))
    data_dir = Path(args.data_dir) if args.data_dir else Path(cfg["data_filepath"])
    out_root = Path(args.out_dir)
    gt_dir = out_root / "gt"
    pred_dir = out_root / "pred"
    mkdir(gt_dir)
    mkdir(pred_dir)

    seed = int(cfg.get("seed", 0))
    split_path = Path(args.split_json) if args.split_json else (Path("assets") / "datainfo" / f"multiple_models_data_split_dict_{seed}.json")
    split_obj = load_json(split_path)
    test_ids = split_obj["test"]
    if not isinstance(test_ids, list):
        raise ValueError(f"split_json['test'] should be list, got {type(test_ids)}")

    # robot states
    rs_path = data_dir / "robot_state.json"
    robot_state_dict = load_json(rs_path)

    # load model
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA not available, fallback to cpu")
        device = "cpu"

    # speed hint on tensor core GPUs
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    model = load_model(cfg, Path(args.ckpt), device=device)
    net = get_net(model)

    # dof: prefer cfg; otherwise infer from first state entry
    dof = int(cfg.get("dof", 0)) if int(cfg.get("dof", 0)) > 0 else None
    if dof is None:
        # infer from first entry length
        k0 = next(iter(robot_state_dict.keys()))
        dof = len(robot_state_dict[k0])
        print(f"[info] infer dof={dof} from robot_state.json")

    # cache grid on GPU for mc method
    grid_cache = None
    if args.method == "mc":
        pts = make_grid(args.res, aabb_min=float(args.aabb_min), aabb_max=float(args.aabb_max))
        grid_cache = torch.from_numpy(pts).to(device)

    # iterate
    total = len(test_ids)
    start = max(int(args.start), 0)
    end = total if args.limit <= 0 else min(total, start + int(args.limit))
    export_ids = test_ids[start:end]

    print(f"[info] Using split: {split_path} 'test' (total={total}), exporting [{start}:{end}) -> {len(export_ids)}")
    print(f"[info] data_dir={data_dir}, out_dir={out_root}")
    print(f"[info] method={args.method}, res={args.res}, iso={args.iso}, AABB=[{args.aabb_min},{args.aabb_max}]")

    for idx in tqdm(export_ids):
        key = int(idx)
        name = f"{key:06d}.ply"

        # --- GT ---
        gt_xyz = load_gt_points(data_dir, key)
        save_ply_xyz(gt_dir / name, gt_xyz)

        # --- state (normalized by /pi, consistent with dataset.py) ---
        entry = robot_state_dict.get(str(key))
        if entry is None:
            # sometimes keys may be int in json (rare)
            entry = robot_state_dict.get(key)
        if entry is None:
            raise KeyError(f"robot_state.json missing key={key}")
        state_raw = parse_state_vector(entry, dof=int(dof))  # raw
        state_norm = (state_raw / math.pi).astype(np.float32, copy=False)

        # --- Pred ---
        pred_n = int(args.pred_npoints) if int(args.pred_npoints) > 0 else int(gt_xyz.shape[0])
        if args.method == "mc":
            pred_xyz = pred_pointcloud_mc(net,
                                          state_norm=state_norm,
                                          res=int(args.res),
                                          iso=float(args.iso),
                                          pred_n=pred_n,
                                          device=device,
                                          aabb_min=float(args.aabb_min),
                                          aabb_max=float(args.aabb_max),
                                          chunk=int(args.chunk),
                                          grid_cache=grid_cache,
                                          keep_largest=bool(args.keep_largest))
        else:
            pred_xyz = pred_pointcloud_neariso(net,
                                               state_norm=state_norm,
                                               pred_n=pred_n,
                                               device=device,
                                               aabb_min=float(args.aabb_min),
                                               aabb_max=float(args.aabb_max),
                                               n_samples=int(args.neariso_samples),
                                               chunk=int(args.chunk),
                                               iso=float(args.iso))

        save_ply_xyz(pred_dir / name, pred_xyz)

    print("[done] Export finished.")


if __name__ == "__main__":
    main()
'''
python eval_export_test_pcd.py \
  --config configs/state_condition/sim_5m_with_base.yaml \
  --ckpt sim_5m_with_base_state-condition_new-global-siren-sdf_43/lightning_logs/version_0/checkpoints/epoch=499-step=141000.ckpt \
  --out-dir eval_out/sim_5m_with_base \
  --device cuda \
  --res 128 \
  --iso 0.0


python eval_export_test_pcd.py \
  --config configs/state_condition/sim_5m_no_base.yaml \
  --ckpt sim_5m_no_base_state-condition_new-global-siren-sdf_43/lightning_logs/version_0/checkpoints/epoch=499-step=141000.ckpt \
  --out-dir eval_out/sim_5m_no_base \
  --device cuda \
  --res 128 \
  --iso 0.0


python eval_export_test_pcd.py \
  --config configs/state_condition/sim_2m_no_base.yaml \
  --ckpt sim_2m_no_base_state-condition_new-global-siren-sdf_6/lightning_logs/version_0/checkpoints/epoch=499-step=70500.ckpt \
  --out-dir eval_out/sim_2m_no_base \
  --device cuda \
  --res 128 \
  --iso 0.0

python eval_export_test_pcd.py \
  --config configs/state_condition/sim_2m_with_base.yaml \
  --ckpt sim_2m_with_base_state-condition_new-global-siren-sdf_6/lightning_logs/version_0/checkpoints/epoch=499-step=70500.ckpt \
  --out-dir eval_out/sim_2m_with_base \
  --device cuda \
  --res 128 \
  --iso 0.0


python eval_export_test_pcd.py \
  --config configs/state_condition/sim_3m_with_base.yaml \
  --ckpt sim_3m_with_base_state-condition_new-global-siren-sdf_6/lightning_logs/version_0/checkpoints/epoch=499-step=70500.ckpt \
  --out-dir eval_out/sim_3m_with_base \
  --device cuda \
  --res 128 \
  --iso 0.0


python eval_export_test_pcd.py \
  --config configs/state_condition/sim_3m_no_base.yaml \
  --ckpt sim_3m_no_base_state-condition_new-global-siren-sdf_6/lightning_logs/version_0/checkpoints/epoch=499-step=70500.ckpt \
  --out-dir eval_out/sim_3m_no_base \
  --device cuda \
  --res 128 \
  --iso 0.0
'''