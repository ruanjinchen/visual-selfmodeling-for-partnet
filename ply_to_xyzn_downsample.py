# -*- coding: utf-8 -*-
"""
ply_to_xyzn_downsample.py

读取指定目录下的 PLY（必须带法向），对每个文件：
- 仅执行：读取 -> 固定点数下采样/补齐 -> 写出 XYZN（x y z nx ny nz）
- **输出到与输入相同的目录**；
- **输出文件名为 `mesh_<index>.xyzn`**，其中 `<index>` 来自输入 PLY 文件名里的数字序号：
  例如：cloud_0.ply → mesh_0.xyzn，cloud_148.ply → mesh_148.xyzn。
- 默认只遍历该目录下一层（非递归）。

依赖：open3d, numpy
用法：
    python ply_to_xyzn_downsample.py <dir> --num 400000 --seed 1
    # 可选：指定匹配模式（默认 cloud_*.ply）
    python ply_to_xyzn_downsample.py <dir> --num 400000 --pattern "cloud_*.ply" --seed 1
"""

import argparse
import re
from pathlib import Path
import numpy as np
import open3d as o3d


def load_ply_with_normals(path: Path):
    pcd = o3d.io.read_point_cloud(str(path))
    if not pcd.has_points():
        raise ValueError(f"[{path.name}] 空点云")
    if not pcd.has_normals():
        raise ValueError(f"[{path.name}] 缺少法向（请先生成带法向的 PLY）")
    pts = np.asarray(pcd.points, dtype=np.float32)
    nrm = np.asarray(pcd.normals, dtype=np.float32)
    if pts.shape[0] != nrm.shape[0]:
        raise ValueError(f"[{path.name}] 点数与法向数不一致：{pts.shape[0]} vs {nrm.shape[0]}")
    # 过滤异常
    ok = np.isfinite(pts).all(axis=1) & np.isfinite(nrm).all(axis=1)
    pts = pts[ok]; nrm = nrm[ok]
    return pts, nrm


def ensure_unit_normals(nrm: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    l = np.linalg.norm(nrm, axis=1, keepdims=True)
    l = np.clip(l, eps, None)
    return nrm / l


def fixed_count_indices(m: int, n_target: int, rng: np.random.Generator) -> np.ndarray:
    """
    返回长度为 n_target 的索引；m>=n 时无放回采样，m<n 时带放回采样。
    这样可保证输出点数**严格等于** n_target。
    """
    if m <= 0:
        return np.zeros((0,), dtype=np.int64)
    if m >= n_target:
        return rng.choice(m, size=n_target, replace=False)
    else:
        return rng.choice(m, size=n_target, replace=True)


def parse_index_from_name(name: str) -> int:
    """
    从文件名中解析尾部数字序号（.ply 之前的连续数字）。
    例如：cloud_0.ply -> 0, cloud_148.ply -> 148。
    若未找到则抛错（严格要求）。
    """
    m = re.search(r'(\d+)(?=\.ply$)', name)
    if not m:
        raise ValueError(f"无法从文件名解析序号：{name}（期望形如 cloud_<index>.ply）")
    return int(m.group(1))


def convert_dir_inplace(dir_path: Path, pattern: str, n_target: int,
                        seed: int = 1, normalize_normals: bool = True,
                        float_fmt: str = "%.8f"):
    files = sorted(dir_path.glob(pattern))
    if not files:
        print(f"[!] 目录 {dir_path} 下未找到匹配 {pattern} 的 PLY 文件")
        return

    print(f"[I] 在 {dir_path} 内就地生成 mesh_<index>.xyzn（固定点数 = {n_target}）")
    for i, f in enumerate(files):
        try:
            idx = parse_index_from_name(f.name)  # 从 PLY 名字提取数字序号
            pts, nrm = load_ply_with_normals(f)
            if normalize_normals:
                nrm = ensure_unit_normals(nrm)

            # 每个文件用稳定子种子（与 index 绑定），确保多次运行可复现
            rng = np.random.default_rng(seed + idx)
            pick = fixed_count_indices(pts.shape[0], int(n_target), rng)
            out = np.hstack([pts[pick], nrm[pick]]).astype(np.float32)  # (N,6)

            out_path = f.with_name(f"mesh_{idx}.xyzn")
            np.savetxt(out_path, out, fmt=float_fmt)
            print(f"[✓] {f.name} -> {out_path.name}  ({out.shape[0]} points)")
        except Exception as e:
            print(f"[X] 跳过 {f.name} ：{e}")


def main():
    ap = argparse.ArgumentParser("In-place PLY(+normals) → mesh_<index>.xyzn 转换（固定点数下采样）")
    ap.add_argument("dir", type=str, help="包含 PLY 的目录（非递归）")
    ap.add_argument("--num", type=int, required=True, help="目标点数（固定）")
    ap.add_argument("--seed", type=int, default=1, help="随机种子（抽样复现；与 index 绑定）")
    ap.add_argument("--pattern", type=str, default="cloud_*.ply", help="文件匹配模式（默认 cloud_*.ply）")
    ap.add_argument("--no-normalize-normals", action="store_true",
                    help="不对法向单位化（默认会单位化）")
    ap.add_argument("--float-fmt", type=str, default="%.8f",
                    help="输出浮点格式（默认 %.8f）")

    args = ap.parse_args()
    dir_path = Path(args.dir).resolve()
    if not dir_path.is_dir():
        raise NotADirectoryError(f"不是有效目录：{dir_path}")

    convert_dir_inplace(
        dir_path=dir_path,
        pattern=args.pattern,
        n_target=int(args.num),
        seed=int(args.seed),
        normalize_normals=not args.no_normalize_normals,
        float_fmt=args.float_fmt
    )


if __name__ == "__main__":
    main()

'''



python ply_to_xyzn_downsample.py data/pliers_2074 \
  --num 20000 \
  --seed 1


python ply_to_xyzn_downsample.py data/scissors_10893 \
  --num 400000 \
  --seed 1


python ply_to_xyzn_downsample.py data/eyeglasses_101863 \
  --num 20000 \
  --seed 1


python ply_to_xyzn_downsample.py data/franka_fr3_normed \
  --num 20000 \
  --seed 1


python ply_to_xyzn_downsample.py data/franka_fr3_original \
  --num 20000 \
  --seed 1


python ply_to_xyzn_downsample.py data/boston_dynamics_spot_original \
  --num 20000 \
  --seed 1
'''